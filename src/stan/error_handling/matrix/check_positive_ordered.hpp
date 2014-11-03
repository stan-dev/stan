#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_POSITIVE_ORDERED_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_POSITIVE_ORDERED_HPP

#include <sstream>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified vector contains
     * only non-negative values and is sorted into increasing order.
     * There may not be duplicate values.  Otherwise, raise a domain
     * error according to the specified policy.
     *
     * @param function
     * @param y Vector to test.
     * @param name
     * @return <code>true</code> if the vector has positive, ordered
     * @return throws if any element in y is nan
     * values.
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

        dom_err(function, name, y[0],
                msg.str(), ", but should be postive.");
      }
      for (size_type n = 1; n < y.size(); n++) {
        if (!(y[n] > y[n-1])) {
          std::ostringstream msg1;
          msg1 << "is not a valid ordered vector."
              << " The element at " << stan::error_index::value + n 
               << " is ";
          std::ostringstream msg2;
          msg2 << ", but should be greater than the previous element, "
               << y[n-1];
          dom_err(function, name, y[n],
                  msg1.str(), msg2.str());
          return false;
        }
      }
      return true;
    }                         

  }
}
#endif
