#ifndef STAN_SERVICES_ARGUMENTS_ARG_TEST_GRAD_ERR_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_TEST_GRAD_ERR_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_test_grad_err: public real_argument {
    public:
      arg_test_grad_err(): real_argument() {
        _name = "error";
        _description = "Error threshold";
        _validity = "0 < error";
        _default = "1e-6";
        _default_value = 1e-6;
        _constrained = true;
        _good_value = 1e-6;
        _bad_value = -1.0;
        _value = _default_value;
      }

      bool is_valid(double value) { return value > 0; }
    };

  }  // services
}  // stan

#endif
