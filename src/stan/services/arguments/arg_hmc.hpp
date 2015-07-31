#ifndef STAN_SERVICES_ARGUMENTS_ARG_HMC_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_HMC_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_engine.hpp>
#include <stan/services/arguments/arg_metric.hpp>
#include <stan/services/arguments/arg_stepsize.hpp>
#include <stan/services/arguments/arg_stepsize_jitter.hpp>

namespace stan {
  namespace services {

    class arg_hmc: public categorical_argument {
    public:
      arg_hmc() {
        _name = "hmc";
        _description = "Hamiltonian Monte Carlo";

        _subarguments.push_back(new arg_engine());
        _subarguments.push_back(new arg_metric());
        _subarguments.push_back(new arg_stepsize());
        _subarguments.push_back(new arg_stepsize_jitter());
      }
    };

  }  // services
}  // stan

#endif

