#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_MATCHING_SIZES_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_MATCHING_SIZES_HPP__

#include <stan/meta/traits.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <string>
#include <typeinfo>

namespace stan {
  namespace math {

    template <typename T_y1, typename T_y2, typename T_result>
    inline bool check_matching_sizes(const char* function,
                                     const T_y1& y1,
                                     const char* name1,
                                     const T_y2& y2,
                                     const char* name2,
                                     T_result* result) {
      if (y1.size() == y2.size())
        return true;

      std::ostringstream msg;
      msg << " (" << typeid(T_y1).name() <<") has size %1% and ("
          << typeid(T_y2).name() << ") has size " << y2.size() 
          << " but they must match in size";
      std::string tmp(msg.str());
      return dom_err(function,
                     typename scalar_type<T_y1>::type(),
                     name1, tmp.c_str(), "", result);
      return false;
    }

  }
}
#endif
