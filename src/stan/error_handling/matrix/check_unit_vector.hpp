#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_UNIT_VECTOR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_UNIT_VECTOR_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified vector is unit vector.
     *
     * <p>The test that the values sum to 1 is done to within the
     * tolerance specified by <code>CONSTRAINT_TOLERANCE</code>.
     *
     * @param function Function name
     * @param name Variable name
     * @param theta Vector to test.
     * @return <code>true</code> if the vector is a unit vector.
     * @return throws if any element in theta is nan
     */
    template <typename T_prob>
    bool check_unit_vector(const std::string& function,
                           const std::string& name,
                           const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      if (theta.size() == 0) {
        dom_err(function, name, 0,
                "is not a valid unit vector. ",
                " elements in the vector.");
      }
      T_prob ssq = theta.squaredNorm();
      if (!(fabs(1.0 - ssq) <= CONSTRAINT_TOLERANCE)) {
        std::stringstream msg;
        msg << "is not a valid unit vector."
            << " The sum of the squares of the elements should be 1, but is ";
        dom_err(function, name, ssq,
                msg.str());
      }
      return true;
    }

  }
}
#endif
