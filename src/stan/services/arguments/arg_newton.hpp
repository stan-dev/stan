#ifndef STAN_SERVICES_ARGUMENTS_ARG_NEWTON_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_NEWTON_HPP

#include <stan/services/arguments/categorical_argument.hpp>

namespace stan {
  namespace services {

    class arg_newton: public categorical_argument {
    public:
      arg_newton() {
        _name = "newton";
        _description = "Newton's method";
      }
    };

  }  // services
}  // stan

#endif

