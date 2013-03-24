#ifndef __STAN__MATH__MATRIX__LOG_DETERMINANT_HPP__
#define __STAN__MATH__MATRIX__LOG_DETERMINANT_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_square.hpp>

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
      stan::math::validate_square(m,"log_determinant");
      return m.colPivHouseholderQr().logAbsDeterminant();
    }
    
  }
}
#endif
