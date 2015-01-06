#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_MATCHING_SIZES_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_MATCHING_SIZES_HPP

#include <stan/meta/traits.hpp>
#include <stan/error_handling/domain_error.hpp>
#include <string>
#include <typeinfo>
#include <stan/error_handling/matrix/check_size_match.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if two structures at the same size.
     *
     * This function only checks the runtime sizes for variables that
     * implement a <code>size()</code> method.  This function does not
     * throw if either variable has nan values.
     * 
     * @tparam T_y1 Type of the first variable
     * @tparam T_y2 Type of the second variable
     *
     * @param function Function name (for error messages)
     * @param name1 First variable name  (for error messages)
     * @param y1 First variable
     * @param name2 Second variable name (for error messages)
     * @param y2 Second variable
     *
     * @return <code>true</code> if the sizes match
     * @throw <code>std::invalid_argument</code> if the sizes do not match
     */
    template <typename T_y1, typename T_y2>
    inline bool check_matching_sizes(const std::string& function,
                                     const std::string& name1,
                                     const T_y1& y1,
                                     const std::string& name2,
                                     const T_y2& y2) {
      check_size_match(function,
                       "size of y1", y1.size(),
                       "size of y2", y2.size());
      return true;
    }

  }
}
#endif
