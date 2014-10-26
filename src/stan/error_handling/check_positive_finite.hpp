#ifndef STAN__ERROR_HANDLING_CHECK_POSITIVE_FINITE_HPP
#define STAN__ERROR_HANDLING_CHECK_POSITIVE_FINITE_HPP

#include <stan/error_handling/check_positive.hpp>
#include <stan/error_handling/check_finite.hpp>

namespace stan {
  namespace math {

    // throws if any element in y is nan
    template <typename T_y, typename T_result>
    inline bool check_positive_finite(const char* function,
                                      const T_y& y,
                                      const char* name,
                                      T_result* result) {
      stan::math::check_positive(function, y, name, result);
      stan::math::check_finite(function, y, name, result);

      return true;
    }

  }
}
#endif
