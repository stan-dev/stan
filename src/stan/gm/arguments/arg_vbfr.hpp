#ifndef __STAN__GM__ARGUMENTS__VBFR__HPP__
#define __STAN__GM__ARGUMENTS__VBFR__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {

  namespace gm {

    class arg_vbfr: public categorical_argument {

    public:

      arg_vbfr() {

        _name = "vbfr";
        _description = "Variational Bayesian inference (full-rank)";

      }

    };

  } // gm

} // stan

#endif

