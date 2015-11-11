#ifndef STAN_SERVICES_SAMPLE_INIT_STATIC_HMC_HPP
#define STAN_SERVICES_SAMPLE_INIT_STATIC_HMC_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/arguments/argument.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {
    namespace sample {

      template<class Sampler>
      bool init_static_hmc(stan::mcmc::base_mcmc* sampler,
                           stan::services::argument* algorithm) {
        stan::services::categorical_argument* hmc
          = dynamic_cast<stan::services::categorical_argument*>
          (algorithm->arg("hmc"));

        stan::services::categorical_argument* base
          = dynamic_cast<stan::services::categorical_argument*>
          (algorithm->arg("hmc")->arg("engine")->arg("static"));

        double epsilon
          = dynamic_cast<stan::services::real_argument*>
          (hmc->arg("stepsize"))->value();
        double epsilon_jitter
          = dynamic_cast<stan::services::real_argument*>
          (hmc->arg("stepsize_jitter"))->value();
        double int_time
          = dynamic_cast<stan::services::real_argument*>(base->arg("int_time"))
          ->value();

        dynamic_cast<Sampler*>(sampler)
          ->set_nominal_stepsize_and_T(epsilon, int_time);
        dynamic_cast<Sampler*>(sampler)->set_stepsize_jitter(epsilon_jitter);

        return true;
      }

    }
  }
}

#endif
