#ifndef STAN_MATH_PRIM_MAT_FUN_COLS_HPP
#define STAN_MATH_PRIM_MAT_FUN_COLS_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T, int R, int C>
    inline
    size_t
    cols(const Eigen::Matrix<T, R, C>& m) {
      return m.cols();
    }

  }
}
#endif
