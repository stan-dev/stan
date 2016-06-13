#ifndef STAN_OLD_SERVICES_ARGUMENTS_ARG_LBFGS_HPP
#define STAN_OLD_SERVICES_ARGUMENTS_ARG_LBFGS_HPP

#include <stan/old_services/arguments/arg_bfgs.hpp>
#include <stan/old_services/arguments/arg_history_size.hpp>

namespace stan {
  namespace services {

    class arg_lbfgs: public arg_bfgs {
    public:
      arg_lbfgs() {
        _name = "lbfgs";
        _description = "LBFGS with linesearch";

        _subarguments.push_back(new arg_history_size());
      }
    };

  }  // services
}  // stan

#endif

