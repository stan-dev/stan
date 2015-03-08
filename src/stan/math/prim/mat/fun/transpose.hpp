#ifndef STAN__MATH__PRIM__MAT__FUN__TRANSPOSE_HPP
#define STAN__MATH__PRIM__MAT__FUN__TRANSPOSE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T, int R, int C>
    Eigen::Matrix<T,C,R>
    inline
    transpose(const Eigen::Matrix<T,R,C>& m) {
      return m.transpose();
    }

  }
}
#endif
