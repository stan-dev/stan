#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_POSITIVE_ORDERED_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_POSITIVE_ORDERED_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/error_handling/dom_err.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified vector contains
     * only non-negative values and is sorted into increasing order.
     * There may not be duplicate values.  Otherwise, raise a domain
     * error according to the specified policy.
     *
     * @param function
     * @param y Vector to test.
     * @param name
     * @param result
     * @return <code>true</code> if the vector has positive, ordered
     * values.
     */
    template <typename T_y, typename T_result>
    bool check_positive_ordered(const char* function,
                                const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                                const char* name,
                                T_result* result) {
      typedef typename Eigen::Matrix<T_y,Eigen::Dynamic,1>::size_type size_type;
      if (y.size() == 0) {
        return true;
      }
      if (y[0] < 0) {
        std::ostringstream stream;
        stream << " is not a valid positive_ordered vector."
               << " The element at " << stan::error_index::value 
               << " is %1%, but should be postive.";
        std::string msg(stream.str());
        return dom_err(function,y[0],name,
                       msg.c_str(),"",
                       result);
      }
      for (size_type n = 1; n < y.size(); n++) {
        if (!(y[n] > y[n-1])) {
          std::ostringstream stream;
          stream << " is not a valid ordered vector."
                 << " The element at " << stan::error_index::value + n 
                 << " is %1%, but should be greater than the previous element, "
                 << y[n-1];
          std::string msg(stream.str());
          return dom_err(function,y[n],name,
                         msg.c_str(),"",
                         result);
        }
      }
      return true;
    }                         

  }
}
#endif
