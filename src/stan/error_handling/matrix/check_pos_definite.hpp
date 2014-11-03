#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is positive definite
     *
     * NOTE: symmetry is NOT checked by this function
     * 
     * @param function
     * @param y Matrix to test.
     * @param name
     * @return <code>true</code> if the matrix is positive definite.
     * @return throws if any element in lower triangular of matrix is nan.
     * @tparam T Type of scalar.
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
        dom_err(function, name, y(0,0), 
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
        dom_err(function, name, y(0,0),
                msg.str());
      }
      return true;
    }

  }
}
#endif
