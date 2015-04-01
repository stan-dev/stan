#ifndef STAN__MATH__PRIM__MAT__FUN__ROWS_HPP
#define STAN__MATH__PRIM__MAT__FUN__ROWS_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T, int R, int C>
    inline
    size_t
    rows(const Eigen::Matrix<T, R, C>& m) {
      return m.rows();
    }

  }
}
#endif
