#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP

#include <string>
#include <typeinfo>

#include <stan/error_handling/dom_err.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace math {

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
     * @param result
     * @return <code>true</code> if the the specified matrix/vector is 
     * of non-zero size
     * @throw std::domain_error if the specified matrix/vector is of
     * non-zero size
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result>
    inline bool check_nonzero_size(const char* function,
                                   const T_y& y,
                                   const char* name,
                                   T_result* result) {
      typedef typename index_type<T_y>::type size_t;
      if (y.size() > 0) 
        return true;

      return dom_err(function, size_t(), 
                     name, " has size %1%, but must have a non-zero size","",
                     result);
    }

  }
}
#endif
