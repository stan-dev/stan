#ifndef STAN__MATH__ERROR_HANDLING__CHECK_CONSISTENT_SIZES_HPP
#define STAN__MATH__ERROR_HANDLING__CHECK_CONSISTENT_SIZES_HPP

#include <stan/math/error_handling/check_consistent_size.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    template <typename T1, typename T2, typename T_result>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const char* name1,
                                       const char* name2,
                                       T_result* result) {
      size_t max_size = std::max(size_of(x1),
                                 size_of(x2));
      return check_consistent_size(max_size,function,x1,name1,result)
        && check_consistent_size(max_size,function,x2,name2,result);
    }

    template <typename T1, typename T2, typename T3, typename T_result>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const T3& x3,
                                       const char* name1,
                                       const char* name2,
                                       const char* name3,
                                       T_result* result) {
      size_t max_size = std::max(size_of(x1),
                                 std::max(size_of(x2),size_of(x3)));
      return check_consistent_size(max_size,function,x1,name1,result)
        && check_consistent_size(max_size,function,x2,name2,result)
        && check_consistent_size(max_size,function,x3,name3,result);
    }
    template <typename T1, typename T2, typename T3, typename T4, typename T_result>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const T3& x3,
                                       const T4& x4,
                                       const char* name1,
                                       const char* name2,
                                       const char* name3,
                                       const char* name4,
                                       T_result* result) {
      size_t max_size = std::max(size_of(x1),
                                 std::max(size_of(x2),
                                          std::max(size_of(x3), size_of(x4))));
      return check_consistent_size(max_size,function,x1,name1,result)
        && check_consistent_size(max_size,function,x2,name2,result)
        && check_consistent_size(max_size,function,x3,name3,result)
        && check_consistent_size(max_size,function,x4,name4,result);
    }

  }
}
#endif
