#ifndef STAN_SERVICES_SAMPLE_INIT_SOFTABS_HPP
#define STAN_SERVICES_SAMPLE_INIT_SOFTABS_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/arguments/argument.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {
    namespace sample {

      template<class Sampler>
      bool init_softabs(stan::mcmc::base_mcmc* sampler,
                        stan::services::argument* algorithm) {
        stan::services::categorical_argument* hmc
          = dynamic_cast<stan::services::categorical_argument*>
          (algorithm->arg("hmc"));

        double alpha
          = dynamic_cast<stan::services::real_argument*>
          (hmc->arg("metric")->arg("softabs")->arg("alpha"))->value();

        dynamic_cast<Sampler*>(sampler)->z()->set_alpha(alpha);

        return true;
      }

    }
  }
}

#endif
