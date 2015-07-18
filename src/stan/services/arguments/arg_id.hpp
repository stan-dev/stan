#ifndef STAN_SERVICES_ARGUMENTS_ARG_ID_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_ID_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_id: public int_argument {
    public:
      arg_id(): int_argument() {
        _name = "id";
        _description = "Unique process identifier";
        _validity = "id > 0";
        _default = "0";
        _default_value = 0;
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

