#ifndef STAN_VARIATIONAL_SUBSAMPLE_HPP
#define STAN_VARIATIONAL_SUBSAMPLE_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_subsample: public bool_argument {
    public:
      arg_variational_subsample(): bool_argument() {
        _name = "subsample";
        _description = "Subsample into minibatch?";
        _validity = "[0, 1]";
        _default = "0";
        _default_value = false;
        _constrained = false;
        _good_value = 1;
        _value = _default_value;
      }
    };

  }  // services
}  // stan

#endif

