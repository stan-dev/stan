#ifndef STAN_SERVICES_ARGUMENTS_ARG_ADAPT_T0_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_ADAPT_T0_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_adapt_t0: public real_argument {
    public:
      arg_adapt_t0(): real_argument() {
        _name = "t0";
        _description = "Adaptation iteration offset";
        _validity = "0 < t0";
        _default = "10";
        _default_value = 10;
        _constrained = true;
        _good_value = 2.0;
        _bad_value = -1.0;
        _value = _default_value;
      }

      bool is_valid(double value) { return value > 0; }
    };

  }  // services
}  // stan

#endif

