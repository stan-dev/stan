#ifndef STAN_SERVICES_ARGUMENTS_ARG_SAVE_ITERATIONS_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_SAVE_ITERATIONS_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_save_iterations: public bool_argument {
    public:
      arg_save_iterations(): bool_argument() {
        _name = "save_iterations";
        _description = "Stream optimization progress to output?";
        _validity = "[0, 1]";
        _default = "0";
        _default_value = false;
        _constrained = false;
        _good_value = 1;
        _value = _default_value;
      }
    };

  }  // services
}  // stan

#endif

