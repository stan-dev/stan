#ifndef STAN__MATH__REV__MAT__FUN__STAN_PRINT_HPP
#define STAN__MATH__REV__MAT__FUN__STAN_PRINT_HPP

#include <stan/math/rev/core.hpp>
#include <ostream>

namespace stan {
  namespace agrad {

    void stan_print(std::ostream* o, const var& x) {
      *o << x.val();
    }

  }
}
#endif
