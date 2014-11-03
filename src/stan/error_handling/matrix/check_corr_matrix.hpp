#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_CORR_MATRIX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_CORR_MATRIX_HPP

#include <sstream>

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/matrix/check_pos_definite.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>


namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * correlation matrix.  A valid correlation matrix is symmetric,
     * has a unit diagonal (all 1 values), and has all values between
     * -1 and 1 (inclussive).  
     *
     * @param function 
     * @param y Matrix to test.
     * @param name 
     * 
     * @return <code>true</code> if the specified matrix is a valid
     * correlation matrix.
     * @return throw if any element in matrix is nan
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_y>
    inline bool check_corr_matrix(const std::string& function,
                                  const std::string& name,
                                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T_y,Dynamic,Dynamic> >::type size_t;

      check_size_match(function, 
                       "Rows of correlation matrix", y.rows(), 
                       "columns of correlation matrix", y.cols());
      
      check_positive(function, "rows", y.rows());

      check_symmetric(function, "y", y);
      
      for (size_t k = 0; k < y.rows(); ++k) {
        if (!(fabs(y(k,k) - 1.0) <= CONSTRAINT_TOLERANCE)) {
          std::ostringstream msg;
          msg << "is not a valid correlation matrix. " 
              << name << "(" << stan::error_index::value + k 
              << "," << stan::error_index::value + k 
              << ") is "; ;
          dom_err(function, name, y(k,k),
                  msg.str(),
                  ", but should be near 1.0");
          return false;
        }
      }
      stan::error_handling::check_pos_definite(function, "y", y);
      return true;
    }

  }
}
#endif
