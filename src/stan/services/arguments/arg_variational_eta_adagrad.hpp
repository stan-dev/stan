#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_ETA_ADAGRAD_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_ETA_ADAGRAD_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_eta_adagrad: public real_argument {
    public:
      arg_variational_eta_adagrad(): real_argument() {
        _name = "eta_adagrad";
        _description = "Stepsize weighting parameter for variational iteration";
        _validity = "0 < eta_adagrad <= 1.0";
        _default = "0.1";
        _default_value = 0.1;
        _constrained = true;
        _good_value = 1.0;
        _bad_value = -1.0;
        _value = _default_value;
      }
      bool is_valid(double value) { return value > 0 && value <= 1.0; }
    };
  }  // services
}  // stan

#endif
