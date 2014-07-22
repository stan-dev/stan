#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_STD_VECTOR_INDEX_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_STD_VECTOR_INDEX_HPP

#include <sstream>
#include <vector>
#include <stan/math/error_handling/dom_err.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified index is valid in std vector
     *
     * @param function
     * @param i is index
     * @param y std vector to test against
     * @param name
     * @param result
     * @return <code>true</code> if the index is a valid in std vector.
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result>
    inline bool check_std_vector_index(const char* function,
                                       size_t i,
                                       const std::vector<T_y>& y,
                                       const char* name,
                                       T_result* result) {
      if ((i > 0) && (i <= static_cast<size_t>(y.size())))
        return true;

      std::ostringstream msg;
      msg << name << " (%1%) must be greater than 0 and less than " 
          << y.size();
      std::string tmp(msg.str());
      return dom_err(function,i,name,
                     tmp.c_str(),"",
                     result);
    }

  }
}
#endif
