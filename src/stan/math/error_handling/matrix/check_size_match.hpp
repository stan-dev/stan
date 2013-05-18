#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SIZE_MATCH_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SIZE_MATCH_HPP__

#include <sstream>
#include <boost/type_traits/common_type.hpp>
#include <stan/math/error_handling/dom_err.hpp>

namespace stan {
  namespace math {

    // FIXME: update warnings
    template <typename T_size1, typename T_size2, typename T_result>
    inline bool check_size_match(const char* function,
                                 T_size1 i,
                                 const char* name_i,
                                 T_size2 j,
                                 const char* name_j,
                                 T_result* result) {
      typedef typename boost::common_type<T_size1,T_size2>::type common_type;
      if (static_cast<common_type>(i) != static_cast<common_type>(j)) {
        std::ostringstream msg;
        msg << name_i << " (%1%) and " 
            << name_j << " (" << j << ") must match in size";
        std::string tmp(msg.str());
        return dom_err(function,i,name_i,
                       tmp.c_str(),"",
                       result);
      }
      return true;
    }

    template <typename T_size1, typename T_size2>
    inline
    bool check_size_match(const char* function,
                          T_size1 i, const char* name_i,
                          T_size2 j, const char* name_j,
                          T_size1* result = 0) {
      return check_size_match<T_size1,T_size2,T_size1>(function,
                                                       i,name_i,
                                                       j,name_j,
                                                       result);
    }

  }
}
#endif
