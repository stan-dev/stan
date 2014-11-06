#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_CONSISTENT_SIZES_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_CONSISTENT_SIZES_HPP

#include <stan/error_handling/scalar/check_consistent_size.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    // NOTE: this will not throw if nan is passed in.
    template <typename T1, typename T2>
    inline bool check_consistent_sizes(const std::string& function,
                                       const std::string& name1,
                                       const T1& x1, 
                                       const std::string& name2,
                                       const T2& x2) {
      size_t max_size = std::max(size_of(x1),
                                 size_of(x2));
      return check_consistent_size(function, name1, x1, max_size)
        && check_consistent_size(function, name2, x2, max_size);
    }

    template <typename T1, typename T2, typename T3>
    inline bool check_consistent_sizes(const std::string& function,
                                       const std::string& name1,
                                       const T1& x1, 
                                       const std::string& name2, 
                                       const T2& x2, 
                                       const std::string& name3, 
                                       const T3& x3) {
      size_t max_size = std::max(size_of(x1),
                                 std::max(size_of(x2),size_of(x3)));
      return check_consistent_size(function, name1, x1, max_size)
        && check_consistent_size(function, name2, x2, max_size)
        && check_consistent_size(function, name3, x3, max_size);
    }
    template <typename T1, typename T2, typename T3, typename T4>
    inline bool check_consistent_sizes(const std::string& function,
                                       const std::string& name1, 
                                       const T1& x1, 
                                       const std::string& name2, 
                                       const T2& x2, 
                                       const std::string& name3, 
                                       const T3& x3, 
                                       const std::string& name4,
                                       const T4& x4) {
      size_t max_size = std::max(size_of(x1),
                                 std::max(size_of(x2),
                                          std::max(size_of(x3), size_of(x4))));
      return check_consistent_size(function, name1, x1, max_size)
        && check_consistent_size(function, name2, x2, max_size)
        && check_consistent_size(function, name3, x3, max_size)
        && check_consistent_size(function, name4, x4, max_size);
    }

  }
}
#endif
