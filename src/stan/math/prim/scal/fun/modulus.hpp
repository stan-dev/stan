#ifndef STAN__MATH__PRIM__SCAL__FUN__MODULUS_HPP
#define STAN__MATH__PRIM__SCAL__FUN__MODULUS_HPP

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
