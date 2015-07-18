#ifndef STAN_SERVICES_ARGUMENTS_ARG_HISTORY_SIZE_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_HISTORY_SIZE_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_history_size: public int_argument {
    public:
      arg_history_size(): int_argument() {
        _name = "history_size";
        _description = "Amount of history to keep for L-BFGS";
        _validity = "0 < history_size";
        _default = "5";
        _default_value = 5;
        _constrained = true;
        _good_value = 2;
        _bad_value = -1;
        _value = _default_value;
      }

      bool is_valid(int value) { return value > 0; }
    };

  }  // services
}  // stan

#endif

