#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP

#include <string>
#include <typeinfo>

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix/vector is of
     * non-zero size. Throws a std:domain_error otherwise. The message
     * will indicate that the variable name "has size 0".
     *
     * NOTE: this will not throw if y contains nan values.
     *
     * @param function
     * @param y matrix/vector to test against
     * @param name
     * @return <code>true</code> if the the specified matrix/vector is 
     * of non-zero size
     * @throw std::domain_error if the specified matrix/vector is of
     * non-zero size
     * @tparam T Type of scalar.
     */
    template <typename T_y>
    inline bool check_nonzero_size(const std::string& function,
                                   const std::string& name,
                                   const T_y& y) {
      using stan::math::index_type;

      typedef typename index_type<T_y>::type size_t;
      if (y.size() > 0) 
        return true;

      dom_err(function, name, size_t(), 
              "has size ", 
              ", but must have a non-zero size");
      return false;
    }

  }
}
#endif
