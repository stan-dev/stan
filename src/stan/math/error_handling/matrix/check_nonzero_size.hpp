#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_NONZERO_SIZE_HPP

#include <stan/meta/traits.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <string>
#include <typeinfo>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix/vector is of non-zero size
     *
     * @param function
     * @param y matrix/vector to test against
     * @param name
     * @param result
     * @return <code>true</code> if the the specified matrix/vector is of non-zero size
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result>
    inline bool check_nonzero_size(const char* function,
                                   const T_y& y,
                                   const char* name,
                                   T_result* result) {
      if (y.size() > 0) 
        return true;

      std::string msg;
      msg += "(";
      msg += typeid(T_y).name();
      msg += ") has size %1%, but must have a non-zero size";
      return dom_err(function,typename T_y::value_type(),
                     name,msg.c_str(),"",
                     result);
    }

  }
}
#endif
