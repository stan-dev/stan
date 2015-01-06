#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP

#include <string>
#include <typeinfo>

#include <stan/error_handling/invalid_argument.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix/vector is of
     * non-zero size. 
     *
     * Throws a std:invalid_argument otherwise. The message
     * will indicate that the variable name "has size 0".
     *
     * NOTE: this will not throw if y contains nan values.
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
    inline bool check_nonzero_size(const std::string& function,
                                   const std::string& name,
                                   const T_y& y) {
      if (y.size() > 0) 
        return true;

      using stan::math::index_type;
      typedef typename index_type<T_y>::type size_t;
      invalid_argument(function, name, size_t(), 
                       "has size ", 
                       ", but must have a non-zero size");
      return false;
    }

  }
}
#endif
