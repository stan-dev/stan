#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_STD_VECTOR_INDEX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_STD_VECTOR_INDEX_HPP

#include <sstream>
#include <vector>
#include <stan/error_handling/scalar/dom_err.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified index is valid in std vector
     *
     * NOTE: this will not throw if y contains nan values.
     *
     * @param function
     * @param i is index
     * @param y std vector to test against
     * @param name
     * @return <code>true</code> if the index is a valid in std vector.
     * @tparam T Type of scalar.
     */
    template <typename T_y>
    inline bool check_std_vector_index(const std::string& function,
                                       const std::string& name,
                                       const std::vector<T_y>& y,
                                       size_t i) {
      if ((i > 0) && (i <= static_cast<size_t>(y.size())))
        return true;

      std::ostringstream msg;
      msg << ") must be greater than 0 and less than " 
          << y.size();
      dom_err(function, name, i,
              "(", msg.str());
      return false;
    }

  }
}
#endif
