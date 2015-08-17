#ifndef STAN_SERVICES_ARGUMENTS_ARG_ADAPT_KAPPA_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_ADAPT_KAPPA_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_adapt_kappa: public real_argument {
    public:
      arg_adapt_kappa(): real_argument() {
        _name = "kappa";
        _description = "Adaptation relaxation exponent";
        _validity = "0 < kappa";
        _default = "0.75";
        _default_value = 0.75;
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

