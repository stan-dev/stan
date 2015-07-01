#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_MEANFIELD_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_MEANFIELD_HPP

#include <stan/services/arguments/categorical_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_meanfield: public categorical_argument {
    public:
      arg_variational_meanfield() {
        _name = "meanfield";
        _description = "mean-field approximation";
      }
    };
  }  // services
}  // stan

#endif

