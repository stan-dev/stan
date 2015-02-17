#ifndef STAN__MATH__REV__MAT__FUN__STAN_PRINT_HPP
#define STAN__MATH__REV__MAT__FUN__STAN_PRINT_HPP

#include <ostream>
#include <stan/math/rev/core/var.hpp>

namespace stan {
  namespace agrad {

    void stan_print(std::ostream* o, const var& x) {
      *o << x.val();
    }

  }
}
#endif
