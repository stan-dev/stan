#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SIMPLEX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SIMPLEX_HPP

#include <sstream>
#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>


namespace stan {

  namespace math {

    /**
     * Return <code>true</code> if the specified vector is simplex.
     * To be a simplex, all values must be greater than or equal to 0
     * and the values must sum to 1.
     *
     * A valid simplex is one where the sum of hte elements is equal
     * to 1.  This function tests that the sum is within the tolerance
     * specified by <code>CONSTRAINT_TOLERANCE</code>. This function
     * only accepts Eigen vectors, statically typed vectors, not
     * general matrices with 1 column.
     *
     * @tparam T_prob Scalar type of the vector
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param theta Vector to test.
     *
     * @return <code>true</code> if the vector is a simplex
     * @throw <code>std::invalid_argument</code> if <code>theta</code>
     *   is a 0-vector.
     * @throw <code>std::domain_error</code> if the vector is not a 
     *   simplex or if any element is <code>NaN</code>.
     */
    template <typename T_prob>
    bool check_simplex(const std::string& function,
                       const std::string& name,
                       const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T_prob,Dynamic,1> >::type size_t;

      check_nonzero_size(function, name, theta);
      if (!(fabs(1.0 - theta.sum()) <= CONSTRAINT_TOLERANCE)) {
        std::stringstream msg;
        T_prob sum = theta.sum();
        msg << "is not a valid simplex.";
        msg.precision(10);
        msg << " sum(" << name << ") = " << sum
            << ", but should be ";
        domain_error(function, name, 1.0,
                msg.str());
         return false;
      }
      for (size_t n = 0; n < theta.size(); n++) {
        if (!(theta[n] >= 0)) {
          std::ostringstream msg;
          msg << "is not a valid simplex. "
                 << name << "[" << n + stan::error_index::value << "]"
                 << " = ";
          domain_error(function, name, theta[n],
                  msg.str(), 
                  ", but should be greater than or equal to 0");
          return false;
        }
      }
      return true;
    }                         

  }
}
#endif
