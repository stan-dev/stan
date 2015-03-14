#ifndef STAN__MATH__PRIM__MAT__ERR__CHECK_POSITIVE_ORDERED_HPP
#define STAN__MATH__PRIM__MAT__ERR__CHECK_POSITIVE_ORDERED_HPP

#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/mat/err/check_ordered.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>

#include <sstream>
#include <string>

namespace stan {

  namespace math {

    /**
     * Return <code>true</code> if the specified vector contains
     * non-negative values and is sorted into strictly increasing
     * order.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Vector to test
     *
     * @return <code>true</code> if the vector is positive, ordered
     * @throw <code>std::domain_error</code> if the vector contains non-positive
     *   values, if the values are not ordered, if there are duplicated
     *   values, or if any element is <code>NaN</code>.
     */
    template <typename T_y>
    bool
    check_positive_ordered(const char* function,
                           const char* name,
                           const Eigen::Matrix<T_y, Eigen::Dynamic, 1>& y) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;

      if (y.size() == 0) {
        return true;
      }
      if (y[0] < 0) {
        std::ostringstream msg;
        msg << "is not a valid positive_ordered vector."
            << " The element at " << stan::error_index::value
            << " is ";
        std::string msg_str(msg.str());
        domain_error(function, name, y[0],
                     msg_str.c_str(), ", but should be postive.");
        return false;
      }
      check_ordered(function, name, y);
      return true;
    }
  }
}
#endif
