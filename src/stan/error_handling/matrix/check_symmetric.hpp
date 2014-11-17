#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SYMMETRIC_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SYMMETRIC_HPP

#include <sstream>

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is symmetric
     * 
     * NOTE: squareness is not checked by this function
     *
     * @param function 
     * @param y Matrix to test.
     * @param name
     * @return <code>true</code> if the matrix is symmetric.
     * @return throws if any element not on the main diagonal is NaN
     * @tparam T Type of scalar.
     */
    template <typename T_y>
    inline bool 
    check_symmetric(const std::string& function,
                    const std::string& name,
                    const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T_y,Dynamic,Dynamic> >::type size_type;

      size_type k = y.rows();
      if (k == 1)
        return true;
      for (size_type m = 0; m < k; ++m) {
        for (size_type n = m + 1; n < k; ++n) {
          if (!(fabs(y(m,n) - y(n,m)) <= CONSTRAINT_TOLERANCE)) {
            std::ostringstream msg1;
            msg1 << "is not symmetric. " 
                    << name << "[" << stan::error_index::value + m << "," 
                    << stan::error_index::value +n << "] is ";
            std::ostringstream msg2;
            msg2 << ", but "
                 << name << "[" << stan::error_index::value +n << "," 
                 << stan::error_index::value + m 
                 << "] element is " << y(n,m);
            dom_err(function, name, y(m,n), 
                    msg1.str(), msg2.str());
            return false;
          }
        }
      }
      return true;
    }

  }
}
#endif
