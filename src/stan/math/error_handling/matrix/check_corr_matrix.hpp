#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_CORR_MATRIX_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_CORR_MATRIX_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/check_positive.hpp>
#include <stan/math/error_handling/matrix/check_pos_definite.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/matrix/constraint_tolerance.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * correlation matrix.  A valid correlation matrix is symmetric,
     * has a unit diagonal (all 1 values), and has all values between
     * -1 and 1 (inclussive).  
     *
     * @param function 
     * @param y Matrix to test.
     * @param name 
     * @param result 
     * 
     * @return <code>true</code> if the specified matrix is a valid
     * correlation matrix.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_y, typename T_result>
    inline bool check_corr_matrix(const char* function,
                                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                                  const char* name,
                                  T_result* result) {
      stan::math::check_size_match(function, 
                                   y.rows(), "Rows of correlation matrix",
                                   y.cols(), "columns of correlation matrix",
                                   result);
      stan::math::check_positive(function, y.rows(), "rows", result);
      stan::math::check_symmetric(function, y, "y", result);
      for (typename Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>::size_type
             k = 0; k < y.rows(); ++k) {
        if (fabs(y(k,k) - 1.0) > CONSTRAINT_TOLERANCE) {
          std::ostringstream message;
          message << " is not a valid correlation matrix. " 
                  << name << "(" << stan::error_index::value + k 
                  << "," << stan::error_index::value + k 
                  << ") is %1%, but should be near 1.0";
          std::string msg(message.str());
          return dom_err(function,y(k,k),name,msg.c_str(),"",result);
        }
      }
      stan::math::check_pos_definite(function, y, "y", result);
      return true;
    }

  }
}
#endif
