#ifndef STAN_OLD_SERVICES_ARGUMENTS_ARG_XHMC_HPP
#define STAN_OLD_SERVICES_ARGUMENTS_ARG_XHMC_HPP

#include <stan/old_services/arguments/categorical_argument.hpp>
#include <stan/old_services/arguments/arg_max_depth.hpp>
#include <stan/old_services/arguments/arg_x_delta.hpp>

namespace stan {
  namespace services {

    class arg_xhmc: public categorical_argument {
    public:
      arg_xhmc() {
        _name = "xhmc";
        _description = "Exhaustive Hamiltonian Monte Carlo";

        _subarguments.push_back(new arg_max_depth());
        _subarguments.push_back(new arg_x_delta());
      }
    };

  }  // services
}  // stan

#endif
