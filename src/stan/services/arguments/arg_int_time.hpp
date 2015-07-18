#ifndef STAN_SERVICES_ARGUMENTS_ARG_INT_TIME_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_INT_TIME_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_int_time: public real_argument {
    public:
      arg_int_time(): real_argument() {
        _name = "int_time";
        _description = "Total integration time for Hamiltonian evolution";
        _validity = "0 < int_time";
        _default = "2 * pi";
        _default_value = 6.28318530717959;
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

