#ifndef STAN__MATH__PRIM__SCAL__FUN__DIVIDE_HPP
#define STAN__MATH__PRIM__SCAL__FUN__DIVIDE_HPP

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
