#ifndef STAN_MCMC_FIXED_PARAM_SAMPLER_HPP
#define STAN_MCMC_FIXED_PARAM_SAMPLER_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>

#include <iostream>
#include <string>

namespace stan {

  namespace mcmc {

    class fixed_param_sampler : public base_mcmc {
    public:
      explicit fixed_param_sampler(std::ostream* o,
                                   std::ostream* e)
        : base_mcmc(o, e) {
        this->name_ = "Fixed Parameter Sampler";
      }

      sample transition(sample& init_sample) {
        return init_sample;
      }
    };

  }  // mcmc

}  // stan

#endif

