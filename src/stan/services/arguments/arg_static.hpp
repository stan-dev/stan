#ifndef STAN_SERVICES_ARGUMENTS_ARG_STATIC_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_STATIC_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_int_time.hpp>

namespace stan {
  namespace services {

    class arg_static: public categorical_argument {
    public:
      arg_static() {
        _name = "static";
        _description = "Static integration time";

        _subarguments.push_back(new arg_int_time());
      }
    };

  }  // services
}  // stan

#endif

