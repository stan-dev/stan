#ifndef STAN_MATH_PRIM_SCAL_FUN_MIN_HPP
#define STAN_MATH_PRIM_SCAL_FUN_MIN_HPP

#include <algorithm>

namespace stan {
  namespace math {

    inline double min(const double a, const double b) {
      return a < b ? a : b;
    }

  }
}

#endif
