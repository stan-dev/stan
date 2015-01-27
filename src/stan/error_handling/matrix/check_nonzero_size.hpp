#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP

#include <string>
#include <typeinfo>

#include <stan/error_handling/invalid_argument.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace math {

    /**
     * Return <code>true</code> if the specified matrix/vector is of
     * non-zero size. 
     *
     * Throws a std:invalid_argument otherwise. The message
     * will indicate that the variable name "has size 0".
     *
     * @tparam T_y Type of container
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Container to test. This will accept matrices and vectors
     *
     * @return <code>true</code> if the the specified matrix/vector is 
     * of non-zero size
     * @throw <code>std::invalid_argument</code> if the specified matrix/vector 
     *   has zero size
     */
    template <typename T_y>
    inline bool check_nonzero_size(const char* function,
                                   const char* name,
                                   const T_y& y) {
      if (y.size() > 0) 
        return true;
      
      invalid_argument(function, name, 0, 
                       "has size ", 
                       ", but must have a non-zero size");
      return false;
    }

  }
}
#endif
