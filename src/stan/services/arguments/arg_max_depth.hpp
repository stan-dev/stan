#ifndef STAN_SERVICES_ARGUMENTS_ARG_MAX_DEPTH_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_MAX_DEPTH_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_max_depth: public int_argument {
    public:
      arg_max_depth(): int_argument() {
        _name = "max_depth";
        _description = "Maximum tree depth";
        _validity = "0 < max_depth";
        _default = "10";
        _default_value = 10;
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

