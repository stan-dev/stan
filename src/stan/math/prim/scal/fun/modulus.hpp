#ifndef STAN_MATH_PRIM_SCAL_FUN_MODULUS_HPP
#define STAN_MATH_PRIM_SCAL_FUN_MODULUS_HPP

#include <cstddef>
#include <cstdlib>

namespace stan {
  namespace math {

    inline int modulus(const int x, const int y) {
      return std::div(x, y).rem;
    }

  }
}

#endif
