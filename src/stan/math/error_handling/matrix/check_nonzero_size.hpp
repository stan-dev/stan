#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP

#include <stan/meta/traits.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <string>
#include <typeinfo>

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
      if (y.size() > 0) 
        return true;

      return dom_err(function,typename T_y::size_type(),
                     name, " has size %1%, but must have a non-zero size","",
                     result);
    }

  }
}
#endif
