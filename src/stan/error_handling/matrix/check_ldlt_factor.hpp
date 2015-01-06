#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_LDLT_FACTOR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_LDLT_FACTOR_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/domain_error.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the argument is a valid 
     * <code>stan::math::LDLT_factor</code>.
     *
     * <code>LDLT_factor</code> can be constructed in an invalid
     * state, so it must be checked. A invalid <code>LDLT_factor</code>
     * is constructed from a non positive definite matrix.
     *
     * @tparam T Type of scalar
     * @tparam R Rows of the matrix
     * @tparam C Columns of the matrix
     * 
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param A <code>stan::math::LDLT_factor</code> to check for validity.
     *
     * @return <code>true</code> if the matrix is positive definite.
     * @return throws <code>std::domain_error</code> the LDLT_factor was
     *   created improperly (A.success() == false)
     */
    template <typename T, int R, int C>
    inline bool check_ldlt_factor(const std::string& function,
                                  const std::string& name,
                                  stan::math::LDLT_factor<T,R,C> &A) {
      if (!A.success()) {
        std::ostringstream msg;
        msg << "is not positive definite. "
            << "last conditional variance is ";
        const T too_small = A.vectorD().tail(1)(0);
        domain_error(function, name, too_small,
                msg.str(), ".");
      }
      return true;
    }

  }
}
#endif
