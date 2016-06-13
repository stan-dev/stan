#ifndef STAN_OLD_SERVICES_ARGUMENTS_ARG_OPTIMIZE_HPP
#define STAN_OLD_SERVICES_ARGUMENTS_ARG_OPTIMIZE_HPP

#include <stan/old_services/arguments/categorical_argument.hpp>
#include <stan/old_services/arguments/arg_optimize_algo.hpp>
#include <stan/old_services/arguments/arg_iter.hpp>
#include <stan/old_services/arguments/arg_save_iterations.hpp>

namespace stan {
  namespace services {

    class arg_optimize: public categorical_argument {
    public:
      arg_optimize() {
        _name = "optimize";
        _description = "Point estimation";

        _subarguments.push_back(new arg_optimize_algo());
        _subarguments.push_back(new arg_iter());
        _subarguments.push_back(new arg_save_iterations());
      }
    };

  }  // services
}  // stan

#endif

