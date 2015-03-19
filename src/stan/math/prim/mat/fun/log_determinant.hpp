#ifndef STAN__MATH__PRIM__MAT__FUN__LOG_DETERMINANT_HPP
#define STAN__MATH__PRIM__MAT__FUN__LOG_DETERMINANT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the log absolute determinant of the specified square matrix.
     *
     * @param m Specified matrix.
     * @return log absolute determinant of the matrix.
     * @throw std::domain_error if matrix is not square.
     */
    template <typename T,int R, int C>
    inline T log_determinant(const Eigen::Matrix<T,R,C>& m) {
      stan::math::check_square("log_determinant", "m", m);
      return m.colPivHouseholderQr().logAbsDeterminant();
    }

  }
}
#endif
