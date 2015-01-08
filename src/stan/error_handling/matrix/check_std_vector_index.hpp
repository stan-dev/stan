#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_STD_VECTOR_INDEX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_STD_VECTOR_INDEX_HPP

#include <sstream>
#include <vector>
#include <stan/error_handling/out_of_range.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified index is valid in std vector
     *
     * This check is 1-indexed by default. This behavior can be changed
     * by setting <code>stan::error_index::value</code>.
     *
     * @tparam T Scalar type
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y <code>std::vector</code> to test
     * @param i Index
     * 
     * @return <code>true</code> if the index is a valid in std vector.
     * @throw <code>std::out_of_range</code> if the index is out of range.
     */
    template <typename T>
    inline bool check_std_vector_index(const std::string& function,
                                       const std::string& name,
                                       const std::vector<T>& y,
                                       size_t i) {
      if ((i >= stan::error_index::value) 
          && (i < y.size() + stan::error_index::value))
        return true;
      
      std::stringstream msg;
      msg << " for " << name;
      out_of_range(function, y.size(), static_cast<int>(i), msg.str());
      return false;
    }

  }
}
#endif
