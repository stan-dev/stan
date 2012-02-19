#ifndef __STAN__MCMC__ADAPTIVE_SAMPLER_HPP__
#define __STAN__MCMC__ADAPTIVE_SAMPLER_HPP__

#include <stan/mcmc/sampler.hpp>

namespace stan {

  namespace mcmc {

    class adaptive_sampler {
    protected:

      // Should we adapt after each call to next()?
      int _adapt;

      // How many calls to next() there have been where _adapt is true
      int _n_adapt_steps;

      // Mean of statistic we want to coerce to some target (if applicable)
      double _mean_stat;

      unsigned int _nfevals;

      unsigned int _n_steps;


    public:
      adaptive_sampler()
        : _adapt(1), 
          _n_adapt_steps(0), 
          _mean_stat(0),
          _nfevals(0), 
          _n_steps(0) {
      }

      virtual ~adaptive_sampler() { }

      /**
       * Set the model real and integer parameters to the specified
       * values.  
       *
       * This method will typically be used to set the parameters
       * by the client of this class after initialization.  
       *
       * @param x Real parameters.
       * @param z Integer parameters.
       */
      virtual void set_params(std::vector<double> x, 
                              std::vector<int> z) = 0;

      /**
       * Return the next sample from this sampler.
       *
       * @return Next sample.
       */
      virtual sample next() = 0;

      /**
       * Find a reasonable initial setting for the adaptable parameters.
       * May not be applicable/implemented for all samplers; the default
       * implementation does nothing.
       */
      virtual void find_reasonable_parameters() { }

      /**
       * Return the values of any tunable parameters for this sampler
       * in the "params" vector.
       * May not be applicable/implemented for all samplers; the default
       * implementation returns an empty vector.
       *
       * @param params Where to store the returned parameters.
       */
      virtual void get_parameters(std::vector<double>& params) { 
        params.resize(0);
      }

      /**
       * Return the value of whatever statistic we're trying to
       * coerce.  For example, if we're trying to set the average
       * acceptance probability of HMC to 0.651 then this will return
       * the realized acceptance probability averaged across all
       * samples so far.
       */
      virtual double mean_stat() { return _mean_stat; }
    };

  }

}

#endif
