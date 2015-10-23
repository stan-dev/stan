#ifndef STAN_SERVICES_ARGUMENTS_ARG_INIT_ALPHA_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_INIT_ALPHA_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_init_alpha: public real_argument {
    public:
      arg_init_alpha(): real_argument() {
        _name = "init_alpha";
        _description = "Line search step size for first iteration";
        _validity = "0 < init_alpha";
        _default = "0.001";
        _default_value = 0.001;
        _constrained = true;
        _good_value = 1.0;
        _bad_value = -1.0;
        _value = _default_value;
      }

      bool is_valid(double value) { return value > 0; }
    };

  }  // services
}  // stan

#endif
