#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_POSITIVE_FINITE_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_POSITIVE_FINITE_HPP

#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>

namespace stan {
  namespace error_handling {

    // throws if any element in y is nan
    template <typename T_y>
    inline bool check_positive_finite(const std::string& function,
                                      const std::string& name,
                                      const T_y& y) {
      stan::error_handling::check_positive(function, name, y);
      stan::error_handling::check_finite(function, name, y);

      return true;
    }

  }
}
#endif
