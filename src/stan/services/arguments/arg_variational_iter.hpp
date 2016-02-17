#ifndef STAN_VARIATIONAL_ITER_HPP
#define STAN_VARIATIONAL_ITER_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_iter: public int_argument {
    public:
      arg_variational_iter(): int_argument() {
        _name = "iter";
        _description = "Maximum number of iterations";
        _validity = "0 < iter";
        _default = "10000";
        _default_value = 10000;
        _constrained = true;
        _good_value = 10000.0;
        _bad_value = -1.0;
        _value = _default_value;
      }

      bool is_valid(int value) {
        return value > 0;
      }
    };

  }  // services

}  // stan

#endif
