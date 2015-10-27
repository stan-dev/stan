#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_ETA_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_ETA_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_eta: public string_argument {
    public:
      arg_variational_eta(): string_argument() {
        _name = "eta";
        _description = "Stepsize scaling parameter for variational inference";
        _validity = "0 < eta <= 1.0";
        _default = "automatically tuned";
        _default_value = "automatically tuned";
        _constrained = false;
        _good_value = "good";
        _value = _default_value;
      }
    };
  }  // services
}  // stan

#endif
