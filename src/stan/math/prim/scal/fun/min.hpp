#ifndef STAN__MATH__PRIM__SCAL__FUN__MIN_HPP
#define STAN__MATH__PRIM__SCAL__FUN__MIN_HPP

#include <algorithm>

namespace stan {
  namespace math {

    inline double min(const double a, const double b) {
      return a < b ? a : b;
    }

  }
}

#endif
