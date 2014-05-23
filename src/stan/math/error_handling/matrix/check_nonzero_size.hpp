#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP__

#include <stdexcept>
#include <sstream>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <iostream>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix/vector is of non-zero size
     *
     * @param function
     * @param y matrix/vector to test against
     * @param name
     * @param result
     * @return <code>true</code> if the the specified matrix/vector is of non-zero size
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result>
    inline bool check_nonzero_size(const char* function,
                                   const T_y& y,
                                   const char* name,
                                   T_result* result) {

      if (y.size() > 0) 
        return true;

      return dom_err(function,y,name,
                     "%1% must have a non-zero size","",
                     result);
    }

  }
}
#endif
