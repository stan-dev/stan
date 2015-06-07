#ifndef STAN_MATH_REV_MAT_FUN_STAN_PRINT_HPP
#define STAN_MATH_REV_MAT_FUN_STAN_PRINT_HPP

#include <stan/math/rev/core.hpp>
#include <ostream>

namespace stan {
  namespace math {

    void stan_print(std::ostream* o, const var& x) {
      *o << x.val();
    }

  }
}
#endif
