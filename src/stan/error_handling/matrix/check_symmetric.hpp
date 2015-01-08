#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SYMMETRIC_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SYMMETRIC_HPP

#include <sstream>

#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is symmetric
     * 
     * The error message is either 0 or 1 indexed, specified by
     * <code>stan::error_index::value</code>.
     *
     * NOTE: squareness is not checked by this function
     *
     * @tparam T_y Type of scalar.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     *
     * @return <code>true</code> if the matrix is symmetric
     * @throw <code>std::domain_error</code> if any element not on the 
     *   main diagonal is <code>NaN</code>
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
            domain_error(function, name, y(m,n), 
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
