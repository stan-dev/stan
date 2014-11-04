#ifndef __STAN__GM__ARGUMENTS__VBMF__HPP__
#define __STAN__GM__ARGUMENTS__VBMF__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {

  namespace gm {

    class arg_vbmf: public categorical_argument {

    public:

      arg_vbmf() {

        _name = "vbmf";
        _description = "Variational Bayesian inference (mean-field)";

      }

    };

  } // gm

} // stan

#endif

