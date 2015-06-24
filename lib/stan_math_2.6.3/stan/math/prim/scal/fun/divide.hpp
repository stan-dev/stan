#ifndef STAN_MATH_PRIM_SCAL_FUN_DIVIDE_HPP
#define STAN_MATH_PRIM_SCAL_FUN_DIVIDE_HPP

#include <cstddef>
#include <cstdlib>

namespace stan {
  namespace math {

    inline int divide(const int x, const int y) {
      return std::div(x, y).quot;
    }

  }
}

#endif
