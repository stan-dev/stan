#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_MATCHING_SIZES_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_MATCHING_SIZES_HPP

#include <stan/meta/traits.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <string>
#include <typeinfo>
#include <stan/error_handling/matrix/check_size_match.hpp>

namespace stan {
  namespace error_handling {

    // NOTE: this will not throw if y1 or y2 contains nan values.
    template <typename T_y1, typename T_y2>
    inline bool check_matching_sizes(const char* function,
                                     const char* name1,
                                     const T_y1& y1,
                                     const char* name2,
                                     const T_y2& y2) {
      check_size_match(function,
                       "size of y1", y1.size(),
                       "size of y2", y2.size());
      return true;
    }

  }
}
#endif
