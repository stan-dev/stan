#ifndef STAN_OLD_SERVICES_ARGUMENTS_ARG_RWM_HPP
#define STAN_OLD_SERVICES_ARGUMENTS_ARG_RWM_HPP

#include <stan/old_services/arguments/categorical_argument.hpp>

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

