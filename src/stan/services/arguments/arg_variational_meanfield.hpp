#ifndef __STAN__SERVICES__ARGUMENTS__VARIATIONAL__MEANFIELD__HPP__
#define __STAN__SERVICES__ARGUMENTS__VARIATIONAL__MEANFIELD__HPP__

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

  } // services

} // stan

#endif

