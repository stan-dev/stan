#ifndef STAN_SERVICES_INIT_INIT_STATIC_UNIFORM_HPP
#define STAN_SERVICES_INIT_INIT_STATIC_UNIFORM_HPP

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/arguments/argument.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {
    namespace init {

      template<class Sampler>
      bool init_static_uniform(stan::mcmc::base_mcmc* sampler,
                               stan::services::argument* algorithm) {
        stan::services::categorical_argument* hmc
          = dynamic_cast<stan::services::categorical_argument*>
          (algorithm->arg("hmc"));

        stan::services::categorical_argument* base
          = dynamic_cast<stan::services::categorical_argument*>
          (algorithm->arg("hmc")->arg("engine")->arg("static_uniform"));

        double epsilon
          = dynamic_cast<stan::services::real_argument*>
          (hmc->arg("stepsize"))->value();
        double epsilon_jitter
          = dynamic_cast<stan::services::real_argument*>
          (hmc->arg("stepsize_jitter"))->value();
        int num_leapfrog
          = dynamic_cast<stan::services::int_argument*>
          (base->arg("num_leapfrog"))->value();

        dynamic_cast<Sampler*>(sampler)->set_nominal_stepsize(epsilon);
        dynamic_cast<Sampler*>(sampler)->set_L(num_leapfrog);
        dynamic_cast<Sampler*>(sampler)->set_stepsize_jitter(epsilon_jitter);

        return true;
      }

    }
  }
}

#endif
