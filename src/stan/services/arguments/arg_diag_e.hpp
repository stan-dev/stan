#ifndef STAN_SERVICES_ARGUMENTS_ARG_DIAG_E_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_DIAG_E_HPP

#include <stan/services/arguments/unvalued_argument.hpp>

namespace stan {
  namespace services {

    class arg_diag_e: public unvalued_argument {
    public:
      arg_diag_e() {
        _name = "diag_e";
        _description = "Euclidean manifold with diag metric";
      }
    };

  }  // services
}  // stan

#endif

