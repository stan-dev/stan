#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_ORDERED_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_ORDERED_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/error_handling/dom_err.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified vector 
     * is sorted into increasing order.
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
    bool check_ordered(const char* function,
                       const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                       const char* name,
                       T_result* result) {
      typedef typename Eigen::Matrix<T_y,Eigen::Dynamic,1>::size_type size_t;
      if (y.size() == 0) {
        return true;
      }
      for (size_t n = 1; n < y.size(); n++) {
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
          return false;
        }
      }
      return true;
    }                         
    template <typename T>
    bool check_ordered(const char* function,
                       const Eigen::Matrix<T,Eigen::Dynamic,1>& y,
                       const char* name,
                       T* result = 0) {
      return check_ordered<T,T>(function,y,name,result);
    }

  }
}
#endif
