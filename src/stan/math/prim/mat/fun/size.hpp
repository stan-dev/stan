#ifndef STAN__MATH__PRIM__MAT__FUN__SIZE_HPP
#define STAN__MATH__PRIM__MAT__FUN__SIZE_HPP

#include <vector>

namespace stan {
  namespace math {

    template <typename T>
    inline int
    size(const std::vector<T>& x) {
      return x.size();
    }

  }
}
#endif
