#ifndef STAN__ERROR_HANDLING_CHECK_POSITIVE_FINITE_HPP
#define STAN__ERROR_HANDLING_CHECK_POSITIVE_FINITE_HPP

#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>

namespace stan {
  namespace error_handling {

    // throws if any element in y is nan
    template <typename T_y, typename T_result>
    inline bool check_positive_finite(const char* function,
                                      const T_y& y,
                                      const char* name,
                                      T_result* result) {
      stan::error_handling::check_positive(function, y, name, result);
      stan::error_handling::check_finite(function, y, name, result);

      return true;
    }

  }
}
#endif
