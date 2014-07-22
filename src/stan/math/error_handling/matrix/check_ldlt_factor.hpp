#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_LDLT_FACTOR_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_LDLT_FACTOR_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the underlying matrix is positive definite
     *
     * @param function
     * @param A 
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is positive definite.
     * @tparam T Type of scalar.
     */
    template <typename T, int R, int C, typename T_result>
    inline bool check_ldlt_factor(const char* function,
                                  stan::math::LDLT_factor<T,R,C> &A,
                                  const char* name,
                                  T_result* result) {
      if (!A.success()) {
        std::ostringstream message;
        message << "underlying matrix is not positive definite. "
                << name << "last conditional variance is %1%.";
        std::string msg(message.str());
        const double too_small = A.vectorD().tail(1)(0);
        return dom_err(function,too_small,name,msg.c_str(),"",result);
      }
      return true;
    }

  }
}
#endif
