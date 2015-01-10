#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SIZE_MATCH_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SIZE_MATCH_HPP

#include <sstream>
#include <boost/type_traits/common_type.hpp>
#include <stan/error_handling/invalid_argument.hpp>
#include <stan/meta/likely.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the provided sizes match.
     *
     * @tparam T_size1 Type of size 1
     * @tparam T_size2 Type of size 2
     *
     * @param function Function name (for error messages)
     * @param name_i Variable name 1 (for error messages)
     * @param i Size 1
     * @param name_j Variable name 2 (for error messages)
     * @param j Size 2
     * 
     * @return <code>true</code> if the sizes match
     * @throw <code>std::invalid_argument</code> if the sizes
     *   do not match
     */
    template <typename T_size1, typename T_size2>
    inline bool check_size_match(const std::string& function,
                                 const std::string& name_i,
                                 T_size1 i,
                                 const std::string& name_j, 
                                 T_size2 j) {
      if (likely(i == static_cast<T_size1>(j)))
        return true;

      std::ostringstream msg;
      msg << ") and " 
          << name_j << " (" << j << ") must match in size";
      invalid_argument(function, name_i, i,
                       "(", msg.str());
      return false;
    }

  }
}
#endif
