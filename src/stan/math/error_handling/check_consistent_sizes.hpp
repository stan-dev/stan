#ifndef __STAN__MATH__ERROR_HANDLING__CHECK_CONSISTENT_SIZES_HPP__
#define __STAN__MATH__ERROR_HANDLING__CHECK_CONSISTENT_SIZES_HPP__

#include <stan/math/error_handling/check_consistent_size.hpp>
#include <stan/math/error_handling/default_policy.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    template <typename T1, typename T2, typename T_result, 
              class Policy>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const char* name1,
                                       const char* name2,
                                       T_result* result,
                                       const Policy&) {
      size_t max_size = std::max(size_of(x1),
                                 size_of(x2));
      return check_consistent_size(max_size,function,x1,name1,result,Policy())
        && check_consistent_size(max_size,function,x2,name2,result,Policy());
    }
    template <typename T1, typename T2, typename T_result>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const char* name1,
                                       const char* name2,
                                       T_result* result) {
      return check_consistent_sizes(function,x1,x2,name1,name2,
                                    result,default_policy());
    }

    template <typename T1, typename T2, typename T3, typename T_result, 
              class Policy>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const T3& x3,
                                       const char* name1,
                                       const char* name2,
                                       const char* name3,
                                       T_result* result,
                                       const Policy&) {
      size_t max_size = std::max(size_of(x1),
                                 std::max(size_of(x2),size_of(x3)));
      return check_consistent_size(max_size,function,x1,name1,result,Policy())
        && check_consistent_size(max_size,function,x2,name2,result,Policy())
        && check_consistent_size(max_size,function,x3,name3,result,Policy());
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
      return check_consistent_sizes(function,x1,x2,x3,name1,name2,name3,
                                    result,default_policy());
    }
    template <typename T1, typename T2, typename T3, typename T4, 
              typename T_result, 
              class Policy>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const T3& x3,
                                       const T4& x4,
                                       const char* name1,
                                       const char* name2,
                                       const char* name3,
                                       const char* name4,
                                       T_result* result,
                                       const Policy&) {
      size_t max_size = std::max(size_of(x1),
                                 std::max(size_of(x2),
                                          std::max(size_of(x3), size_of(x4))));
      return check_consistent_size(max_size,function,x1,name1,result,Policy())
        && check_consistent_size(max_size,function,x2,name2,result,Policy())
        && check_consistent_size(max_size,function,x3,name3,result,Policy())
        && check_consistent_size(max_size,function,x4,name4,result,Policy());
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
      return check_consistent_sizes(function,x1,x2,x3,x4,
                                    name1,name2,name3,name4,
                                    result,default_policy());
    }

  }
}
#endif
