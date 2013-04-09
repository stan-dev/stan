#ifndef __STAN__MATH__MATRIX__DIVIDE_HPP__
#define __STAN__MATH__MATRIX__DIVIDE_HPP__

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    /**
     * Return specified matrix divided by specified scalar.
     * @tparam R Row type for matrix.
     * @tparam C Column type for matrix.
     * @param m Matrix.
     * @param c Scalar.
     * @return Matrix divided by scalar.
     */
    template <int R, int C>
    inline
    Eigen::Matrix<double,R,C>
    divide(const Eigen::Matrix<double,R,C>& m,
           double c) {
      return m / c;
    }

  }
}
#endif
