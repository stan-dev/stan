#ifndef STAN_OLD_SERVICES_ARGUMENTS_ARG_DIAGNOSE_HPP
#define STAN_OLD_SERVICES_ARGUMENTS_ARG_DIAGNOSE_HPP

#include <stan/old_services/arguments/categorical_argument.hpp>
#include <stan/old_services/arguments/arg_test.hpp>

namespace stan {
  namespace services {

    class arg_diagnose: public categorical_argument {
    public:
      arg_diagnose() {
        _name = "diagnose";
        _description = "Model diagnostics";

        _subarguments.push_back(new arg_test());
      }
    };

  }  // services
}  // stan

#endif

