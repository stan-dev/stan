#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_POS_SEMIDEFINITE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_POS_SEMIDEFINITE_HPP

#include <sstream>
#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_size.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is positive definite
     *
     * @tparam T_y scalar type of the matrix
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     *
     * @return <code>true</code> if the matrix is positive semi-definite.
     * @throw <code>std::invalid_argument</code> if the matrix is not square
     *   or if the matrix has 0 size.
     * @throw <code>std::domain_error</code> if the matrix is not symmetric,
     *   or if it is not positive semi-definite,
     *   or if any element of the matrix is <code>NaN</code>.
     */
    template <typename T_y>
    inline bool 
    check_pos_semidefinite(const char* function,
                           const char* name,
                           const Eigen::Matrix<T_y, Eigen::Dynamic, Eigen::Dynamic>& y) {
      check_symmetric(function, name, y);
      check_positive_size(function, name, "rows", y.rows());
      
      if (y.rows() == 1 && !(y(0,0) >= 0.0))
        domain_error(function, name, y, "is not positive semi-definite: ");

      using Eigen::LDLT;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      LDLT<Matrix<T_y,Dynamic,Dynamic> > cholesky = y.ldlt();
      if (cholesky.info() != Eigen::Success || (cholesky.vectorD().array() < 0.0).any())
        domain_error(function, name, y, "is not positive semi-definite:\n");
      check_not_nan(function, name, y);
      return true;
    }

  }
}
#endif
