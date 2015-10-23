#ifndef STAN_SERVICES_ARGUMENTS_ARG_THIN_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_THIN_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_thin: public int_argument {
    public:
      arg_thin(): int_argument() {
        _name = "thin";
        _description = "Period between saved samples";
        _validity = "0 < thin";
        _default = "1";
        _default_value = 1;
        _constrained = true;
        _good_value = 2.0;
        _bad_value = -1.0;
        _value = _default_value;
      }

      bool is_valid(int value) { return value > 0; }
    };

  }  // services
}  // stan

#endif

