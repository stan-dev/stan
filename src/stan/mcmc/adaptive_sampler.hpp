#ifndef __STAN__MCMC__ADAPTIVE_SAMPLER_HPP__
#define __STAN__MCMC__ADAPTIVE_SAMPLER_HPP__

#include "stan/mcmc/sampler.hpp"


namespace stan {

  namespace mcmc {

    class adaptive_sampler : public sampler {
    protected:
      // Should we adapt after each call to next()?
      int _adapt;
      // How many calls to next() there have been where _adapt is true
      int _n_adapt_steps;
      // Mean of statistic we want to coerce to some target (if applicable)
      double _mean_stat;

    public:
      adaptive_sampler() : _adapt(1), _n_adapt_steps(0), _mean_stat(0) { }

      /**
       * Turn on parameter adaptation.
       */
      virtual void adapt_on() { _adapt = 1; }

      /**
       * Turn off parameter adaptation.
       */
      virtual void adapt_off() { _adapt = 0; }

      /**
       * Return whether or not parameter adaptation is on.
       *
       * @return Whether or not parameter adaptation is on.
       */
      int adapting() { return _adapt; }

      /**
       * Return how many iterations parameter adaptation has happened for.
       *
       * @return How many iterations parameter adaptation has happened for.
       */
      int n_adapt_steps() { return _n_adapt_steps; }

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
