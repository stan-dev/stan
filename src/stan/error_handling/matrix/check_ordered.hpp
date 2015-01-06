#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_ORDERED_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_ORDERED_HPP

#include <sstream>

#include <stan/error_handling/domain_error.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified vector is sorted into
     * strictly increasing order.
     *
     * @tparam T_y Type of scalar
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Vector to test
     *
     * @return <code>true</code> if the vector is ordered
     * @throw <code>std::domain_error</code> if the vector elements are 
     *   not ordered, if there are duplicated
     *   values, or if any element is <code>NaN</code>.
     */
    template <typename T_y>
    bool check_ordered(const std::string& function,
                       const std::string& name,
                       const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T_y,Dynamic,1> >::type size_t;

      if (y.size() == 0) 
        return true;
      
      for (size_t n = 1; n < y.size(); n++) {
        if (!(y[n] > y[n-1])) {
          std::ostringstream msg1;
          msg1 << "is not a valid ordered vector."
               << " The element at " << stan::error_index::value + n 
               << " is ";
          std::ostringstream msg2;
          msg2 << ", but should be greater than the previous element, "
               << y[n-1];
          domain_error(function, name, y[n],
                  msg1.str(), msg2.str());
          return false;
        }
      }
      return true;
    }  
    
    /**
     * Return <code>true</code> if the specified vector is sorted into
     * strictly increasing order.
     *
     * @tparam T_y Type of scalar
     * 
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y <code>std::vector</code> to test
     *
     * @return <code>true</code> if the vector is ordered
     * @throw <code>std::domain_error</code> if the vector elements are 
     *   not ordered, if there are duplicated
     *   values, or if any element is <code>NaN</code>.
     */
    template <typename T_y>
    bool check_ordered(const std::string& function,
                       const std::string& name,
                       const std::vector<T_y>& y) {
      if (y.size() == 0) 
        return true;

      for (int n = 1; n < y.size(); n++) {
        if (!(y[n] > y[n-1])) {
          std::ostringstream msg1;
          msg1 << "is not a valid ordered vector."
               << " The element at " << stan::error_index::value + n 
               << " is ";
          std::ostringstream msg2;
          msg2 << ", but should be greater than the previous element, "
               << y[n-1];
          domain_error(function, name, y[n],
                  msg1.str(), msg2.str());
          return false;
        }
      }
      return true;
    }                         

  }
}
#endif
