#ifndef __STAN__MCMC__CHAINS_HPP__
#define __STAN__MCMC__CHAINS_HPP__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <fstream>
#include <cstdlib>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/accumulators/statistics/p_square_quantile.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/covariance.hpp>
#include <boost/accumulators/statistics/variates/covariate.hpp>


#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/additive_combine.hpp>

#include <stan/math/matrix.hpp>
#include <stan/math/matrix/variance.hpp>
#include <stan/prob/autocorrelation.hpp>
#include <stan/prob/autocovariance.hpp>

#include <stan/io/stan_csv_reader.hpp>

namespace stan {  

  namespace mcmc {
    
    /**
     * An <code>mcmc::chains</code> object stores parameter names and
     * dimensionalities along with samples from multiple chains.
     *
     * <p><b>Synchronization</b>: For arbitrary concurrent use, the
     * read and write methods need to be read/write locked.  Multiple
     * writers can be used concurrently if they write to different
     * chains.  Readers for single chains need only be read/write locked
     * with writers of that chain.  For reading across chains, full
     * read/write locking is required.  Thus methods will be classified
     * as global or single-chain read or write methods.
     *
     * <p><b>Storage Order</b>: Storage is column/last-index major.
     */
    template <typename RNG = boost::random::ecuyer1988>
    class chains {
    private:
      Eigen::Matrix<std::string, Eigen::Dynamic, 1> param_names_;
      Eigen::Matrix<Eigen::MatrixXd, Eigen::Dynamic, 1> samples_;
      Eigen::VectorXi warmup_;
      
      double mean(const Eigen::VectorXd& x) {
        return (x.array() / x.size()).sum();
      }

      double variance(const Eigen::VectorXd& x) {
        double m = mean(x);
        return ((x.array() - m) / std::sqrt((x.size() - 1.0))).square().sum();
      }

      double sd(const Eigen::VectorXd& x) {
        return std::sqrt(variance(x));
      }


      double covariance(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        if (x.rows() != y.rows())
          std::cerr << "warning: covariance of different length chains";
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::variance;
        using boost::accumulators::tag::covariance;
        using boost::accumulators::tag::covariate1;

        accumulator_set<double, stats<covariance<double, covariate1> > > acc;
  
        int M = std::min(x.size(), y.size());
        for (int i = 0; i < M; i++)
          acc(x(i), boost::accumulators::covariate1=y(i));
  
        return boost::accumulators::covariance(acc) * M / (M-1);
      }

      double correlation(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        if (x.rows() != y.rows())
          std::cerr << "warning: covariance of different length chains";
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::variance;
        using boost::accumulators::tag::covariance;
        using boost::accumulators::tag::covariate1;

        accumulator_set<double, stats<variance, covariance<double, covariate1> > > acc_xy;
        accumulator_set<double, stats<variance> > acc_y;
  
        int M = std::min(x.size(), y.size());
        for (int i = 0; i < M; i++) {
          acc_xy(x(i), boost::accumulators::covariate1=y(i));
          acc_y(y(i));
        }
  
        double cov = boost::accumulators::covariance(acc_xy);
        if (cov > -1e-8 && cov < 1e-8)
          return cov;
        return cov / std::sqrt(boost::accumulators::variance(acc_xy) * boost::accumulators::variance(acc_y));
      }
      
      double quantile(const Eigen::VectorXd& x, const double prob) {
        using boost::accumulators::accumulator_set;
        using boost::accumulators::left;
        using boost::accumulators::quantile_probability;
        using boost::accumulators::right;
        using boost::accumulators::stats;
        using boost::accumulators::tag::tail;
        using boost::accumulators::tag::tail_quantile;
        double M = x.rows();
        //size_t cache_size = std::min(prob, 1-prob)*M + 2;
        size_t cache_size = M;

        if (prob < 0.5) {
          accumulator_set<double, stats<tail_quantile<left> > > 
            acc(tail<left>::cache_size = cache_size);
          for (int i = 0; i < M; i++)
            acc(x(i));
          return boost::accumulators::quantile(acc, quantile_probability=prob);
        } 
        accumulator_set<double, stats<tail_quantile<right> > > 
          acc(tail<right>::cache_size = cache_size);
        for (int i = 0; i < M; i++)
          acc(x(i));
        return boost::accumulators::quantile(acc, quantile_probability=prob);
      }

      Eigen::VectorXd quantiles(const Eigen::VectorXd& x, const Eigen::VectorXd& probs) {
        using boost::accumulators::accumulator_set;
        using boost::accumulators::left;
        using boost::accumulators::quantile_probability;
        using boost::accumulators::right;
        using boost::accumulators::stats;
        using boost::accumulators::tag::tail;
        using boost::accumulators::tag::tail_quantile;
        double M = x.rows();

        //size_t cache_size = M/2 + 2;
        size_t cache_size = M;///2 + 2;

        accumulator_set<double, stats<tail_quantile<left> > > 
          acc_left(tail<left>::cache_size = cache_size);
        accumulator_set<double, stats<tail_quantile<right> > > 
          acc_right(tail<right>::cache_size = cache_size);
  
        for (int i = 0; i < M; i++) {
          acc_left(x(i));
          acc_right(x(i));
        }

        Eigen::VectorXd q(probs.size());  
        for (int i = 0; i < probs.size(); i++) {
          if (probs(i) < 0.5) 
            q(i) = boost::accumulators::quantile(acc_left, quantile_probability=probs(i));
          else
            q(i) = boost::accumulators::quantile(acc_right, quantile_probability=probs(i));
        }
        return q;
      }

      Eigen::VectorXd autocorrelation(const Eigen::VectorXd& x) {
        std::vector<double> ac;
        std::vector<double> sample(x.size());
        for (int i = 0; i < x.size(); i++)
          sample[i] = x(i);
        stan::prob::autocorrelation(sample, ac);

        Eigen::VectorXd ac2(ac.size());
        for (std::vector<double>::size_type i = 0; i < ac.size(); i++)
          ac2(i) = ac[i];
        return ac2;
      }

      Eigen::VectorXd autocovariance(const Eigen::VectorXd& x) {
        std::vector<double> ac;
        std::vector<double> sample(x.size());
        for (int i = 0; i < x.size(); i++)
          sample[i] = x(i);
        stan::prob::autocovariance(sample, ac);

        Eigen::VectorXd ac2(ac.size());
        for (std::vector<double>::size_type i = 0; i < ac.size(); i++)
          ac2(i) = ac[i];
        return ac2;
      }

      /** 
       * Returns the effective sample size for the specified parameter
       * across all kept samples.
       *
       * The implementation matches BDA3's effective size description.
       * 
       * Current implementation takes the minimum number of samples
       * across chains as the number of samples per chain.
       * 
       * @param VectorXd 
       * @param Dynamic 
       * @param samples 
       * 
       * @return 
       */
      double effective_sample_size(const Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> &samples) {
        int chains = samples.size();
  
        // need to generalize to each jagged samples per chain
        int n_samples = samples(0).size();
        for (int chain = 1; chain < chains; chain++) {
          n_samples = std::min(n_samples, int(samples(chain).size()));
        }

        Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> acov(chains);
        for (int chain = 0; chain < chains; chain++) {
          acov(chain) = autocovariance(samples(chain));
        }
  
        Eigen::VectorXd chain_mean(chains);
        Eigen::VectorXd chain_var(chains);
        for (int chain = 0; chain < chains; chain++) {
          double n_kept_samples = num_kept_samples(chain);
          chain_mean(chain) = mean(samples(chain));
          chain_var(chain) = acov(chain)(0)*n_kept_samples/(n_kept_samples-1);
        }
      
        double mean_var = mean(chain_var);
        double var_plus = mean_var*(n_samples-1)/n_samples;
        if (chains > 1) 
          var_plus += variance(chain_mean);
        Eigen::VectorXd rho_hat_t(n_samples);
        rho_hat_t.setZero();
        double rho_hat = 0;
        int max_t = 0;
        for (int t = 1; (t < n_samples && rho_hat >= 0); t++) {
          Eigen::VectorXd acov_t(chains);
          for (int chain = 0; chain < chains; chain++) {
            acov_t(chain) = acov(chain)(t);
          }
          rho_hat = 1 - (mean_var - mean(acov_t)) / var_plus;
          if (rho_hat >= 0)
            rho_hat_t(t) = rho_hat;
          max_t = t;
        }
        double ess = chains * n_samples;
        if (max_t > 1) {
          ess /= 1 + 2 * rho_hat_t.sum();
        }
        return ess;
      }

      /** 
       * Return the split potential scale reduction (split R hat)
       * for the specified parameter.
       *
       * Current implementation takes the minimum number of samples
       * across chains as the number of samples per chain.
       * 
       * @param VectorXd 
       * @param Dynamic 
       * @param samples 
       * 
       * @return 
       */
      double split_potential_scale_reduction(const Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> &samples) {
        int chains = samples.size();
        int n_samples = samples(0).size();
        for (int chain = 1; chain < chains; chain++) {
          n_samples = std::min(n_samples, int(samples(chain).size()));
        }
        if (n_samples % 2 == 1)
          n_samples--;
        int n = n_samples / 2;

        Eigen::VectorXd split_chain_mean(2*chains);
        Eigen::VectorXd split_chain_var(2*chains);
  
        for (int chain = 0; chain < chains; chain++) {
          split_chain_mean(2*chain) = mean(samples(chain).topRows(n));
          split_chain_mean(2*chain+1) = mean(samples(chain).bottomRows(n));
    
          split_chain_var(2*chain) = variance(samples(chain).topRows(n));
          split_chain_var(2*chain+1) = variance(samples(chain).bottomRows(n));
        }


        double var_between = n * variance(split_chain_mean);
        double var_within = mean(split_chain_var);
      
        // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
        return sqrt((var_between/var_within + n-1)/n);
      }
      
    public:
      chains(const Eigen::Matrix<std::string, Eigen::Dynamic, 1>& param_names) 
        : param_names_(param_names) { }
      
      chains(const stan::io::stan_csv& stan_csv) 
        : param_names_(stan_csv.header) {
        if (stan_csv.samples.rows() > 0)
          add(stan_csv);
      }
      
      inline const int num_chains() {
        return samples_.size();
      }
      
      inline const int num_params() { 
        return param_names_.size();
      }

      const Eigen::Matrix<std::string, Eigen::Dynamic, 1>& param_names() {
        return param_names_;
      }

      const std::string& param_name(int j) {
        return param_names_(j);
      }

      const int index(const std::string& name) {
        int index = -1;
        for (int i = 0; i < param_names_.size(); i++)
          if (param_names_(i) == name)
            return i;
        return index;
      }

      void set_warmup(const int chain, const int warmup) {
        warmup_(chain) = warmup;
      }
      
      void set_warmup(const int warmup) {
        warmup_.setConstant(warmup);
      }
      
      const Eigen::VectorXi& warmup() {
        return warmup_;
      }
      
      const int warmup(const int chain) {
        return warmup_(chain);
      }

      const int num_samples(const int chain) {
        return samples_(chain).rows();
      }

      const int num_samples() {
        int n = 0;
        for (int chain = 0; chain < num_chains(); chain++)
          n += num_samples(chain);
        return n;
      }

      const int num_kept_samples(const int chain) {
        return num_samples(chain) - warmup(chain);
      }
      
      const int num_kept_samples() {
        int n = 0;
        for (int chain = 0; chain < num_chains(); chain++)
          n += num_kept_samples(chain);
        return n;
      }
      
      void add(const int chain,
               const Eigen::MatrixXd& sample) {
        if (sample.cols() != num_params())
          throw std::invalid_argument("add(chain,sample): number of columns in sample does not match chains");
        if (num_chains() == 0 || chain >= num_chains()) {
          int n = num_chains();
    
          // Need this block for Windows. conservativeResize does not keep the references.
          Eigen::Matrix<Eigen::MatrixXd, Eigen::Dynamic, 1> samples_copy(num_chains());
          Eigen::VectorXi warmup_copy(num_chains());
          for (int i = 0; i < n; i++) {
            samples_copy(i) = samples_(i);
            warmup_copy(i) = warmup_(i);
          }
    
          samples_.resize(chain+1);
          warmup_.resize(chain+1);
          for (int i = 0; i < n; i++) {
            samples_(i) = samples_copy(i);
            warmup_(i) = warmup_copy(i);
          }
          for (int i = n; i < chain+1; i++) {
            samples_(i) = Eigen::MatrixXd(0, num_params());
            warmup_(i) = 0;
          }
        }
        int row = samples_(chain).rows();
        Eigen::MatrixXd new_samples(row+sample.rows(), num_params());
        new_samples << samples_(chain), sample;
        samples_(chain) = new_samples;
      }

      void add(const Eigen::MatrixXd& sample) {
        if (sample.rows() == 0)
          return;
        if (sample.cols() != num_params())
          throw std::invalid_argument("add(sample): number of columns in sample does not match chains");
        add(num_chains(), sample);
      }

      void add(const stan::io::stan_csv& stan_csv) {
        if (stan_csv.header.size() != num_params())
          throw std::invalid_argument("add(stan_csv): number of columns in sample does not match chains");
        if (!param_names_.cwiseEqual(stan_csv.header).all()) {
          throw std::invalid_argument("add(stan_csv): header does not match chain's header");
        }
        add(stan_csv.samples);
        if (stan_csv.metadata.save_warmup)
          set_warmup(num_chains()-1, stan_csv.metadata.num_warmup);
      }
      
      Eigen::VectorXd samples(const int chain, const int index) {
        return samples_(chain).col(index).bottomRows(num_kept_samples(chain));
      }
      
      Eigen::VectorXd samples(const int index) {
        Eigen::VectorXd s(num_kept_samples());
        int start = 0;
        for (int chain = 0; chain < num_chains(); chain++) {
          int n = num_kept_samples(chain);
          s.middleRows(start,n) = samples_(chain).col(index).bottomRows(n);
          start += n;
        }
        return s;
      }

      Eigen::VectorXd samples(const int chain, const std::string& name) {
        return samples(chain,index(name));
      }
      
      Eigen::VectorXd samples(const std::string& name) {
        return samples(index(name));
      }
      
      double mean(const int chain, const int index) {
        return mean(samples(chain,index));
      }
      
      double mean(const int index) {
        return mean(samples(index));
      }

      double mean(const int chain, const std::string& name) {
        return mean(chain, index(name));
      }
      
      double mean(const std::string& name) {
        return mean(index(name));
      }

      double sd(const int chain, const int index) { 
        return sd(samples(chain,index));
      }
      
      double sd(const int index) {
        return sd(samples(index));
      }

      double sd(const int chain, const std::string& name) { 
        return sd(chain, index(name));
      }
      
      double sd(const std::string& name) {
        return sd(index(name));
      }
      
      double variance(const int chain, const int index) {
        return variance(samples(chain,index));
      }
      
      double variance(const int index) {   
        return variance(samples(index));
      }

      double variance(const int chain, const std::string& name) {
        return variance(chain, index(name));
      }
      
      double variance(const std::string& name) {   
        return variance(index(name));
      }

      double covariance(const int chain, const int index1, const int index2) {
        return covariance(samples(chain,index1), samples(chain,index2));
      }
      
      double covariance(const int index1, const int index2) {
        return covariance(samples(index1), samples(index2));
      }

      double covariance(const int chain, const std::string& name1, const std::string& name2) {
        return covariance(chain, index(name1), index(name2));
      }
      
      double covariance(const std::string& name1, const std::string& name2) {
        return covariance(index(name1), index(name2));
      }

      double correlation(const int chain, const int index1, const int index2) {
        return correlation(samples(chain,index1),samples(chain,index2));
      }
      
      double correlation(const int index1, const int index2) {
        return correlation(samples(index1),samples(index2));
      }

      double correlation(const int chain, const std::string& name1, const std::string& name2) {
        return correlation(chain, index(name1), index(name2));
      }
      
      double correlation(const std::string& name1, const std::string& name2) {
        return correlation(index(name1), index(name2));
      }

      double quantile(const int chain, const int index, const double prob) {
        return quantile(samples(chain,index), prob);
      }
      
      double quantile(const int index, const double prob) {
        return quantile(samples(index), prob);
      }

      double quantile(const int chain, const std::string& name, const double prob) {
        return quantile(chain, index(name), prob);
      }
      
      double quantile(const std::string& name, const double prob) {
        return quantile(index(name), prob);
      }

      Eigen::VectorXd quantiles(const int chain, const int index, const Eigen::VectorXd& probs) {
        return quantiles(samples(chain,index), probs);
      }

      Eigen::VectorXd quantiles(const int index, const Eigen::VectorXd& probs) {
        return quantiles(samples(index), probs);
      }

      Eigen::VectorXd quantiles(const int chain, 
                                const std::string& name, const Eigen::VectorXd& probs) {
        return quantiles(chain, index(name), probs);
      }

      Eigen::VectorXd quantiles(const std::string& name, const Eigen::VectorXd& probs) {
        return quantiles(index(name), probs);
      }
      
      Eigen::Vector2d central_interval(const int chain, const int index, const double prob) {
        double low_prob = (1-prob)/2;
        double high_prob = 1-low_prob;
  
        Eigen::Vector2d interval;
        interval << quantile(chain,index,low_prob), quantile(chain,index,high_prob);
        return interval;
      }

      Eigen::Vector2d central_interval(const int index, const double prob) {
        double low_prob = (1-prob)/2;
        double high_prob = 1-low_prob;
  
        Eigen::Vector2d interval;
        interval << quantile(index,low_prob), quantile(index,high_prob);
        return interval;
      }

      Eigen::Vector2d central_interval(const int chain, 
                                       const std::string& name, const double prob) {
        return central_interval(chain, index(name), prob);
      }

      Eigen::Vector2d central_interval(const std::string& name, const double prob) {
        return central_interval(index(name), prob);
      }

      Eigen::VectorXd autocorrelation(const int chain, const int index) {
        return autocorrelation(samples(chain, index));
      }

      Eigen::VectorXd autocorrelation(const int chain, const std::string& name) {
        return autocorrelation(chain, index(name));
      }
      
      Eigen::VectorXd autocovariance(const int chain, const int index) {
        return autocovariance(samples(chain,index));
      }

      Eigen::VectorXd autocovariance(const int chain, const std::string& name) {
        return autocovariance(chain, index(name));
      }

      // FIXME: reimplement using autocorrelation.
      double effective_sample_size(const int index) {
        Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> samples(num_chains());
        for (int chain = 0; chain < num_chains(); chain++) {
          samples(chain) = this->samples(chain, index);
        }
        return effective_sample_size(samples);
      }

      double effective_sample_size(const std::string& name) {
        return effective_sample_size(index(name));
      }
      
      double split_potential_scale_reduction(const int index) {  
        Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> samples(num_chains());
        for (int chain = 0; chain < num_chains(); chain++) {
          samples(chain) = this->samples(chain, index);
        }
        return split_potential_scale_reduction(samples);
      }
      
      double split_potential_scale_reduction(const std::string& name) {  
        return split_potential_scale_reduction(index(name));
      }
    };

  }
}

#endif
