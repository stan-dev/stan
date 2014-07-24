#ifndef STAN__MATH__ERROR_HANDLING__CHECK_CONSISTENT_SIZE_HPP
#define STAN__MATH__ERROR_HANDLING__CHECK_CONSISTENT_SIZE_HPP

#include <stan/math/error_handling/dom_err.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    template <typename T, typename T_result>
    inline bool check_consistent_size(size_t max_size,
                                      const char* function,
                                      const T& x,
                                      const char* name,
                                      T_result* result) {
      size_t x_size = stan::size_of(x);
      if (is_vector<T>::value && x_size == max_size)
        return true;
      if (!is_vector<T>::value && x_size == 1)
        return true;
      return dom_err(function,x_size,name,
                     " (max size) is %1%, but must be consistent, 1 or max=",max_size,
                     result);
    }

  }
}
#endif
