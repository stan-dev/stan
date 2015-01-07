#ifndef __STAN__GM__ARGUMENTS__VARIATIONAL__MEANFIELD__HPP__
#define __STAN__GM__ARGUMENTS__VARIATIONAL__MEANFIELD__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {

  namespace gm {

    class arg_variational_meanfield: public categorical_argument {

    public:

      arg_variational_meanfield() {

        _name = "meanfield";
        _description = "mean-field approximation";

      }

    };

  } // gm

} // stan

#endif

