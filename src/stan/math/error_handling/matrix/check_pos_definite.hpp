#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/matrix/constraint_tolerance.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is positive definite
     *
     * NOTE: symmetry is NOT checked by this function
     * 
     * @param function
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is positive definite.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings (message has (0,0) item)
    template <typename T_y, typename T_result>
    inline bool check_pos_definite(const char* function,
                                   const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                                   const char* name,
                                   T_result* result) {
      if (y.rows() == 1 && y(0,0) <= CONSTRAINT_TOLERANCE) {
        std::ostringstream message;
        message << name << " is not positive definite. " 
                << name << "(0,0) is %1%.";
        std::string msg(message.str());
        return dom_err(function,y(0,0),name,msg.c_str(),"",result);
      }
      Eigen::LDLT< Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> > cholesky 
        = y.ldlt();
      if(cholesky.info() != Eigen::Success || 
         cholesky.isNegative() ||
         (cholesky.vectorD().array() <= CONSTRAINT_TOLERANCE).any()) {
        std::ostringstream message;
        message << name << " is not positive definite. " 
                << name << "(0,0) is %1%.";
        std::string msg(message.str());
        return dom_err(function,y(0,0),name,msg.c_str(),"",result);
      }
      return true;
    }

  }
}
#endif
