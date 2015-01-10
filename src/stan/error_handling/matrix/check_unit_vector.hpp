#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_UNIT_VECTOR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_UNIT_VECTOR_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/error_handling/matrix/check_nonzero_size.hpp>

namespace stan {

  namespace math {

    /**
     * Return <code>true</code> if the specified vector is unit vector.
     *
     * A valid unit vector is one where the square of the elements
     * summed is equal to 1. This function tests that the sum is within the
     * tolerance specified by <code>CONSTRAINT_TOLERANCE</code>.  This
     * function only accepts Eigen vectors, statically typed vectors,
     * not general matrices with 1 column.
     *
     * @tparam T_prob Scalar type of the vector
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param theta Vector to test.
     *
     * @return <code>true</code> if the vector is a unit vector.
     * @throw <code>std::invalid_argument</code> if <code>theta</code>
     *   is a 0-vector.
     * @throw <code>std::domain_error</code> if the vector is not a unit
     *   vector or if any element is <code>NaN</code>.
     */
    template <typename T_prob>
    bool check_unit_vector(const std::string& function,
                           const std::string& name,
                           const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      check_nonzero_size(function, name, theta);
      T_prob ssq = theta.squaredNorm();
      if (!(fabs(1.0 - ssq) <= CONSTRAINT_TOLERANCE)) {
        std::stringstream msg;
        msg << "is not a valid unit vector."
            << " The sum of the squares of the elements should be 1, but is ";
        domain_error(function, name, ssq, msg.str());
      }
      return true;
    }

  }
}
#endif
