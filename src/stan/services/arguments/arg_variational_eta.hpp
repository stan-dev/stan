#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_ETA_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_ETA_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_eta: public real_argument {
    public:
      arg_variational_eta(): real_argument() {
        _name = "eta";
        _description = "Stepsize scaling parameter for variational inference";
        _validity = "0 < eta";
        _default = "1.0";
        _default_value = 1.0;
        _constrained = true;
        _good_value = 1.0;
        _bad_value = -1.0;
        _value = _default_value;
      }
      bool is_valid(double value) { return value > 0; }
    };
  }  // services
}  // stan

#endif
