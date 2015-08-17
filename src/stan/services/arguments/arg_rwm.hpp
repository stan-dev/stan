#ifndef STAN_SERVICES_ARGUMENTS_ARG_RWM_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_RWM_HPP

#include <stan/services/arguments/categorical_argument.hpp>

namespace stan {
  namespace services {

    class arg_rwm: public categorical_argument {
    public:
      arg_rwm() {
        _name = "rwm";
        _description = "Random Walk Metropolis Monte Carlo";
      }
    };

  }  // services
}  // stan

#endif

