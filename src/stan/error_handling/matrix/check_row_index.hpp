#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_ROW_INDEX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_ROW_INDEX_HPP

#include <sstream>
#include <stan/error_handling/out_of_range.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified index is a valid row of the matrix
     *
     * This check is 1-indexed by default. This behavior can be changed
     * by setting <code>stan::error_index::value</code>.
     *
     * @tparam T Scalar type
     * @tparam R Compile time rows
     * @tparam C Compile time columns
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     * @param i is index
     * 
     * @return <code>true</code> if the index is a valid row index in the matrix
     * @throw <code>std::out_of_range</code> if the index is out of range.
     */
    template <typename T_y, int R, int C>
    inline bool check_row_index(const std::string& function,
                                const std::string& name, 
                                const Eigen::Matrix<T_y,R,C>& y, 
                                size_t i) {
      if ((i >= stan::error_index::value) 
          && (i < static_cast<size_t>(y.rows()) + stan::error_index::value))
        return true;
      
      std::stringstream msg;
      msg << " for rows of " << name;
      out_of_range(function, 
                   static_cast<int>(y.rows()),
                   static_cast<int>(i), 
                   msg.str());
      return false;
    }

  }
}
#endif
