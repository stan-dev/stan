#ifndef STAN__MATH__PRIM__SCAL__FUN__MAX_HPP
#define STAN__MATH__PRIM__SCAL__FUN__MAX_HPP

#include <algorithm>

namespace stan {
  namespace math {

    inline double max(const double a, const double b) {
      return a > b ? a : b;
    }

  }
}

#endif
