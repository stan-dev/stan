#ifndef STAN_MCMC_CHAINSET_HPP
#define STAN_MCMC_CHAINSET_HPP

#include <stan/io/stan_csv_reader.hpp>
#include <stan/math/prim.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_rhat.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/accumulators/statistics/p_square_quantile.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/covariance.hpp>
#include <boost/accumulators/statistics/variates/covariate.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <cstdlib>

namespace stan {
namespace mcmc {
  using Eigen::Dynamic;

  /**
   * An <code>mcmc::chainset</code> object manages the post-warmup draws
   * across a set of MCMC chains, which all have the same number or samples.
   *
   * <p><b>Storage Order</b>: Storage is column/last-index major.
   */
  template <typename Unused = void*>
  class chainset {
  private:
    size_t num_samples_;
    std::vector<std::string> param_names_;
    std::vector<Eigen::MatrixXd> chains_;

    static size_t thinned_samples(const stan::io::stan_csv& stan_csv) {
      size_t thinned_samples = stan_csv.metadata.num_samples;
      if (stan_csv.metadata.thin > 0) {
        thinned_samples = thinned_samples / stan_csv.metadata.thin;
      }
      return thinned_samples;
    }

    static bool is_valid(const stan::io::stan_csv& stan_csv) {
      if (stan_csv.header.empty()) return false;
      if (stan_csv.samples.size() == 0) return false;
      if (stan_csv.samples.rows() != thinned_samples(stan_csv)) return false;
      return true;
    }

    /**
     * Process first chain: record header, thinned samples,
     * add samples to vector chains.
     */
    void initFromStanCsv(const stan::io::stan_csv& stan_csv) {
      if (!is_valid(stan_csv)) {
        throw std::invalid_argument("Invalid sample");
      }
      if (chains_.size() > 0) {
        throw std::invalid_argument("Cannot re-initialize chains object");
      }
      param_names_ = stan_csv.header;
      num_samples_ = thinned_samples(stan_csv);
      chains_.push_back(stan_csv.samples);
    }

    /**
     * Process next chain: validate size, shape, column names,
     * append to vector chains.
     */
    void add(const stan::io::stan_csv& stan_csv) {
      if (!is_valid(stan_csv)) {
        throw std::invalid_argument("Invalid sample");
      }
      if (stan_csv.header.size() != num_params()) {
        throw std::invalid_argument(
                                    "Error add(stan_csv): number of columns in"
                                    " sample does not match first chain");
      }
      if (thinned_samples(stan_csv) != num_samples_) {
        throw std::invalid_argument(
                                    "Error add(stan_csv): number of sampling draws in"
                                    " sample does not match first chain");
      }
      for (int i = 0; i < num_params(); i++) {
        if (param_names_[i] != stan_csv.header[i]) {
          std::stringstream ss;
          ss << "Error add(stan_csv): header " << param_names_[i]
             << " does not match chain's header (" << stan_csv.header[i] << ")";
          throw std::invalid_argument(ss.str());
        }
      }
      chains_.push_back(stan_csv.samples);
    }

  public:
    explicit chainset(const stan::io::stan_csv& stan_csv) {
      initFromStanCsv(stan_csv);
    }

    explicit chainset(const std::vector<stan::io::stan_csv>& stan_csv) {
      if (stan_csv.empty())
        return;
      initFromStanCsv(stan_csv[0]);
      for (size_t i = 1; i < stan_csv.size(); ++i) {
        add(stan_csv[i]);
      }
    }

    inline int num_chains() const { return chains_.size(); }

    inline int num_params() const { return param_names_.size(); }

    inline int num_samples() const { return num_samples_; }

    const std::vector<std::string>& param_names() const { return param_names_; }

    const std::string& param_name(int j) const { return param_names_[j]; }

    int index(const std::string& name) const {
      auto it = std::find(param_names_.begin(), param_names_.end(), name);
      if (it != param_names_.end()) {
        return std::distance(param_names_.begin(), it);
      }
      return -1;
    }

    // checks that sample values across all chains are finite and non-identical
    // run this check before calculating between-chain, w/in chain
    // variance, correlation, and covariance.
    bool is_finite_and_varies(const int index) const {
      Eigen::VectorXd draw_zeros = Eigen::VectorXd::Zero(num_chains());
      for (std::size_t i = 0; i < num_chains(); ++i) {
        Eigen::VectorXd draws = chains_[i].col(index);
        draw_zeros(i) = draws(0);
        for (int j = 0; j < num_samples(); ++j) {
          if (!std::isfinite(draws(j)))
            return false;
        }
        if (draws.isApproxToConstant(draws(0))) {
          return false;
        }
      }
      if (num_chains() > 1 && draw_zeros.isApproxToConstant(draw_zeros(0))) {
        return false;
      }
      return true;
    }

    Eigen::VectorXd samples(const int colIndex) const {
      Eigen::VectorXd result(chains_.size() * num_samples_);
      for (int i = 0; i < chains_.size(); ++i) {
        result.segment(i * num_samples_, num_samples_) = chains_[i].col(colIndex);
      }
      return result;
    }

    Eigen::VectorXd samples(const int chain, const int index) const {
      if (index < 0 || index >= param_names_.size()
          || chain < 0 || chain >= num_chains()) {
        throw std::out_of_range("Index out of range");
      }
      return chains_[chain].col(index);
    }

    Eigen::VectorXd samples(const int chain, const std::string& name) const {
      return samples(chain, index(name));
    }

    Eigen::VectorXd samples(const std::string& name) const {
      return samples(index(name));
    }

    // double mean(const int index) const { return samples(index).mean(); }

    // double mean(const std::string& name) const { return mean(index(name)); }

    // double sd(const int index) const { return sd(samples(index)); }

    // double sd(const std::string& name) const { return sd(index(name)); }

    // double variance(const int index) const { return variance(samples(index)); }

    // double variance(const std::string& name) const {
    //   return variance(index(name));
    // }

    // double covariance(const int index1, const int index2) const {
    //   return covariance(samples(index1), samples(index2));
    // }

    // double covariance(const std::string& name1, const std::string& name2) const {
    //   return covariance(index(name1), index(name2));
    // }

    // double correlation(const int index1, const int index2) const {
    //   return correlation(samples(index1), samples(index2));
    // }

    // double correlation(const std::string& name1, const std::string& name2) const {
    //   return correlation(index(name1), index(name2));
    // }

    // double quantile(const int index, const double prob) const {
    //   return quantile(samples(index), prob);
    // }

    // double quantile(const std::string& name, const double prob) const {
    //   return quantile(index(name), prob);
    // }

    // Eigen::VectorXd quantiles(int index, const Eigen::VectorXd& probs) const {
    //   return quantiles(samples(index), probs);
    // }

    // Eigen::VectorXd quantiles(const std::string& name,
    //                           const Eigen::VectorXd& probs) const {
    //   return quantiles(index(name), probs);
    // }

    // Eigen::Vector2d central_interval(int index, double prob) const {
    //   double low_prob = (1 - prob) / 2;
    //   double high_prob = 1 - low_prob;

    //   Eigen::Vector2d interval;
    //   interval << quantile(index, low_prob), quantile(index, high_prob);
    //   return interval;
    // }

    // Eigen::Vector2d central_interval(const std::string& name, double prob) const {
    //   return central_interval(index(name), prob);
    // }

    // Eigen::VectorXd autocorrelation(int chain, const std::string& name) const {
    //   return autocorrelation(chain, index(name));
    // }

    // Eigen::VectorXd autocovariance(int chain, const std::string& name) const {
    //   return autocovariance(chain, index(name));
    // }

    std::pair<double, double>
    split_rank_normalized_rhat(const int index) const {
      if (!is_finite_and_varies(index)) {
        return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
      }
      return analyze::compute_split_rank_normalized_rhat(chains_, index);
    }

    std::pair<double, double>
    split_rank_normalized_rhat(const std::string& name) const {
      return split_rank_normalized_rhat(index(name));
    }

    std::pair<double, double>
    split_rank_normalized_ess(const int index) const {
      if (!is_finite_and_varies(index) || num_samples() < 3) {
        return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
      }
      return analyze::compute_split_rank_normalized_ess(chains_, index);
    }

    std::pair<double, double>
    split_rank_normalized_ess(const std::string& name) const {
      return split_rank_normalized_ess(index(name));
    }

  };

}  // namespace mcmc
}  // namespace stan

#endif
