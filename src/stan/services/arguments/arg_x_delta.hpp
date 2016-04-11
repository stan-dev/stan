#ifndef STAN_SERVICES_ARGUMENTS_ARG_X_DELTA_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_X_DELTA_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_x_delta: public real_argument {
    public:
      arg_x_delta(): real_argument() {
        _name = "x_delta";
        _description = "Exhaustion tolerance";
        _validity = "0 < x_delta";
        _default = "0.1";
        _default_value = 0.1;
        _constrained = true;
        _good_value = 0.1;
        _bad_value = -0.1;
        _value = _default_value;
      }

      bool is_valid(double value) { return value > 0; }
    };

  }  // services
}  // stan

#endif
