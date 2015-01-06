#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_POSITIVE_ORDERED_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_POSITIVE_ORDERED_HPP

#include <sstream>
#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/matrix/check_ordered.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace error_handling {

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
    bool check_positive_ordered(const std::string& function, 
                                const std::string& name,
                                const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T_y,Dynamic,1> >::type size_type;
      if (y.size() == 0) {
        return true;
      }
      if (y[0] < 0) {
        std::ostringstream msg;
        msg << "is not a valid positive_ordered vector."
            << " The element at " << stan::error_index::value 
            << " is ";

        domain_error(function, name, y[0],
                msg.str(), ", but should be postive.");
      }
      check_ordered(function, name, y);
      return true;
    }                         

  }
}
#endif
