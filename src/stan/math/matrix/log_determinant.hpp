#ifndef STAN__MATH__MATRIX__LOG_DETERMINANT_HPP
#define STAN__MATH__MATRIX__LOG_DETERMINANT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>

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
      stan::math::check_square("log_determinant(%1%)",m,"m",(double*)0);
      return m.colPivHouseholderQr().logAbsDeterminant();
    }
    
  }
}
#endif
