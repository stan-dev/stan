#ifndef STAN__SERVICES__ARGUMENTS__VARIATIONAL_ETA_STEPSIZE__HPP
#define STAN__SERVICES__ARGUMENTS__VARIATIONAL_ETA_STEPSIZE__HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_eta_stepsize: public real_argument {

    public:

      arg_variational_eta_stepsize(): real_argument() {
        _name = "eta_stepsize";
        _description = "Step size weighting parameter for variational iteration";
        _validity = "0 < init_stepsize <= 1.0";
        _default = "0.1";
        _default_value = 0.1;
        _constrained = true;
        _good_value = 1.0;
        _bad_value = -1.0;
        _value = _default_value;
      };

      bool is_valid(double value) { return value > 0 && value <= 1.0; }

    };

  } // services

} // stan

#endif
