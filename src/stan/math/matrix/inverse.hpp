#ifndef __STAN__MATH__MATRIX__INVERSE_HPP__
#define __STAN__MATH__MATRIX__INVERSE_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the inverse of the specified matrix.
     * @param m Specified matrix.
     * @return Inverse of the matrix.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    inverse(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      validate_square(m,"matrix inverse");
      return m.inverse();
    }

  }
}
#endif
