#ifndef __STAN__MCMC__MCMC_OUTPUT_HPP__
#define __STAN__MCMC__MCMC_OUTPUT_HPP__

#include <vector>
#include <stdexcept>
#include <stan/math/matrix.hpp>
/*#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>*/


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
       * Implementation matches Keeny's variogram method (neff2() in 
       * ess-functions.txt as posted on the dev list).
       * 
       * @return a vector of the effective size
       */
      std::vector<double> effectiveSize() {
        std::vector<double> ess;
        
        return ess;
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
      
      // Want this to be private. How to test?
      double getVarPlus() {
        std::vector<double> psi;
        //using namespace boost::accumulators;
        //accumulator_set<double, stats<tag::mean, tag::moment<2> > > acc;

        // mean of each sample_[ii]
        for (size_t chain = 0; chain < nChains_; chain++) {
          psi.push_back(stan::math::mean(samples_[chain]));
          std::cout << "samples mean: " << stan::math::mean(samples_[chain]) << std::endl;
        }
        double psi_bar = stan::math::mean(psi);
        double B = 0;        
        
        //B <- n/(m-1)*sum((psi-psi.bar)^2)   
        //s <- numeric(m)
        //for (i in 1:m) s[i] <- 1/(n-1)*sum((x[,i]-psi[i])^2)
        //W <- mean(s)
        //var.plus <- (n-1)*W/n + B/n

        return 0;
      }


    private:
      std::vector< std::vector<double> > samples_;
      size_t nChains_;
      size_t nSamplesPerChain_;

      

    };
   
  }
}
#endif
