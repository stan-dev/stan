#ifndef STAN_SERVICES_ARGUMENTS_ARG_STEPSIZE_JITTER_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_STEPSIZE_JITTER_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_stepsize_jitter: public real_argument {
    public:
      arg_stepsize_jitter(): real_argument() {
        _name = "stepsize_jitter";
        _description = "Uniformly random jitter of the stepsize, in percent";
        _validity = "0 <= stepsize_jitter <= 1";
        _default = "0";
        _default_value = 0.0;
        _constrained = true;
        _good_value = 0.5;
        _bad_value = -1.0;
        _value = _default_value;
      }

      bool is_valid(double value) { return 0 <= value && value <= 1; }
    };

  }  // services
}  // stan

#endif
