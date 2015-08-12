#ifndef STAN_MATH_PRIM_MAT_FUN_SIZE_HPP
#define STAN_MATH_PRIM_MAT_FUN_SIZE_HPP

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
