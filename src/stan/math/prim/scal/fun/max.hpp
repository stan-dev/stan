#ifndef STAN_MATH_PRIM_SCAL_FUN_MAX_HPP
#define STAN_MATH_PRIM_SCAL_FUN_MAX_HPP

#include <algorithm>

namespace stan {
  namespace math {

    inline double max(const double a, const double b) {
      return a > b ? a : b;
    }

  }
}

#endif
