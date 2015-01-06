#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is positive definite.
     *
     * Note: symmetry is NOT checked by this function
     * 
     * @tparam T_y Type of scalar of the matrix
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     * 
     * @return <code>true</code> if the matrix is positive definite
     * @throw <code>std::domain_error</code> if the matrix is not
     *   positive definite or if any element is <code>NaN</code>.
     */
    // FIXME: update warnings (message has (0,0) item)
    template <typename T_y>
    inline bool check_pos_definite(const std::string& function,
                                   const std::string& name,
                                   const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      if (y.rows() == 1 && !(y(0,0) > CONSTRAINT_TOLERANCE)) {
        std::ostringstream msg;
        msg << "is not positive definite. " 
                << name << "(0,0) is ";
        domain_error(function, name, y(0,0), 
                msg.str());
      }
      for (int i = 0; i < y.size(); ++i)
        check_not_nan(function, name, y(i));

      Eigen::LDLT< Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> > cholesky 
        = y.ldlt();
      if (cholesky.info() != Eigen::Success
          || !cholesky.isPositive()
          || (cholesky.vectorD().array() <= CONSTRAINT_TOLERANCE).any()) {
        std::ostringstream msg;
        msg << "is not positive definite. " 
                << name << "(0,0) is ";
        domain_error(function, name, y(0,0),
                msg.str());
      }
      return true;
    }

  }
}
#endif
