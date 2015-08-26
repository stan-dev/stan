#ifndef STAN_VARIATIONAL_TUNING_ITER_HPP
#define STAN_VARIATIONAL_TUNING_ITER_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_tuning_iter: public int_argument {
    public:
      arg_variational_tuning_iter(): int_argument() {
        _name = "tuning_iter";
        _description = "Maximum number of hyperparameter tuning iterations";
        _validity = "0 < tuning_iter";
        _default = "50";
        _default_value = 50;
        _constrained = true;
        _good_value = 2.0;
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
