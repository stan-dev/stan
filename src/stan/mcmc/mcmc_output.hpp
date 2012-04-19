
#ifndef __STAN__MCMC__MCMC_OUTPUT_HPP__
#define __STAN__MCMC__MCMC_OUTPUT_HPP__

#include <vector>
#include <stdexcept>
#include <stan/math/matrix.hpp>

/*#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>*/


#include <iostream>

namespace stan {
  namespace mcmc {
    
    /** 
     * mcmc_output
     *   
     * currently assuming we do everything in batch.
     */
    class mcmc_output {
    public:
      /** 
       * Default constructor
       * 
       */
      mcmc_output() :
        nChains_(0), nSamplesPerChain_(0) {
      }
      
      /** 
       * Construct mcmc_output with samples.
       * 
       * @param samples 
       */
      mcmc_output(std::vector< std::vector<double> > samples) : 
        samples_(samples),
        nChains_(samples.size()),
        nSamplesPerChain_(0) {
        // check chain
        if (nChains_ > 1) {
          nSamplesPerChain_ = samples_[0].size();
          for (size_t chain = 1; chain < nSamplesPerChain_; chain++) {
            if (samples_[chain].size() != nSamplesPerChain_) {
              throw std::domain_error("chain length must be the same");
            }
          }
        }
      }
      
      /** 
       * Add samples
       * 
       * @param chain 
       */
      void add_chain(std::vector<double> chain) {
        // check chain
        if (nChains_ > 0) {
          if (chain.size() != nSamplesPerChain_) {
            throw std::domain_error("chain length must be the same");
          }
        }
        samples_.push_back(chain);
        nChains_++;
        nSamplesPerChain_ = chain.size();
      }

      /** 
       * Effective sample size calculation.
       *
       * Implementation matches BDA3's effective size implementation
       * 
       * @return the effective size
       */
      double effectiveSize() {
        size_t m = nChains_;
        size_t n = nSamplesPerChain_;
        
        std::vector<double> chain_mean;
        std::vector<double> chain_var;
        for (size_t chain = 0; chain < m; chain++) {
          chain_mean.push_back(stan::math::mean(samples_[chain]));
          chain_var.push_back(stan::math::variance(samples_[chain]));
        }
        double var_plus = stan::math::mean(chain_var)*(n-1)/n + stan::math::variance(chain_mean);
        
        std::vector<double> rho_hat_t;
        double rho_hat = 0;
        for (size_t t = 1; t < n & rho_hat >= 0; t++) {
          double variogram = 0;
          for (size_t chain = 0; chain < m; chain++) {
            for (size_t ii = 0; ii < n-t; ii++) {
              double diff = samples_[chain][ii] - samples_[chain][ii+t]; 
              variogram += diff * diff;
            }
          }
          variogram /= m * (n-t);
          rho_hat = 1 - variogram / (2 * var_plus);
          if (rho_hat >= 0)
            rho_hat_t.push_back(rho_hat);
        }        
      
        double ess = m*n;
        if (rho_hat_t.size() > 0) {
          ess /= 1 + 2 * stan::math::sum(rho_hat_t);
        }
        return ess;
      }
      
      double splitRHat() {
        //using namespace boost::accumulators;
        size_t m = nChains_*2;
        size_t n = nSamplesPerChain_/2;
        
        std::vector<double> chain_mean;
        std::vector<double> chain_var;
        std::vector<double> chain(n);
        for (size_t ii = 0; ii < m; ii++) {
          if (ii % 2 == 0)
            chain.assign(samples_[ii/2].begin(), samples_[ii/2].begin()+n);
          else
            chain.assign(samples_[ii/2].end()-n, samples_[ii/2].end());
          chain_mean.push_back(stan::math::mean(chain));
          chain_var.push_back(stan::math::variance(chain));
        }
        double var_between = n * stan::math::variance(chain_mean);
        double var_within = stan::math::mean(chain_var);
        
        // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
        return sqrt((var_between/var_within + n-1)/n);
      }

      size_t nChains() {
        return nChains_;
      }
      size_t nSamplesPerChain() {
        return nSamplesPerChain_;
      }
      
      friend std::ostream& operator<<(std::ostream& os, const mcmc_output& mcmc_output) {
        os << mcmc_output.samples_.size() << " chains  ";
        if (mcmc_output.samples_.size() > 0)
          os << ", " << mcmc_output.samples_[0].size() << " samples per chain";
        os << std::endl;
        return os;
      }

    private:
      std::vector< std::vector<double> > samples_;
      size_t nChains_;
      size_t nSamplesPerChain_;

      

    };
   
  }
}
#endif
