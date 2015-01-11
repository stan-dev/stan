#ifndef STAN__MATH__MATRIX__TRANSPOSE_HPP
#define STAN__MATH__MATRIX__TRANSPOSE_HPP

#include <stan/math/matrix/Eigen.hpp>

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
