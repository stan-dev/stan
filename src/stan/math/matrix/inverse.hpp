#ifndef STAN__MATH__MATRIX__INVERSE_HPP
#define STAN__MATH__MATRIX__INVERSE_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>

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
      stan::math::check_square("inverse(%1%)",m,"m",(double*)0);
      return m.inverse();
    }

  }
}
#endif
