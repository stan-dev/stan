#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_COLUMN_INDEX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_COLUMN_INDEX_HPP

#include <sstream>
#include <stan/error_handling/out_of_range.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified index is a valid
     * column of the matrix.
     *
     * By default, this is a 1-indexed check (as opposed to
     * 0-indexed). Behavior can be changed by setting 
     * <code>stan::error_index::value</code>. This function will
     * throw an <code>std::out_of_range</code> exception if
     * the index is out of bounds.
     * 
     * @tparam T_y Type of scalar.
     * @tparam R Number of rows of the matrix
     * @tparam C Number of columns of the matrix
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix 
     * @param i Index to check
     * 
     * @return <code>true</code> if the index is a valid column index 
     *   of the matrix.
     * @throw std::out_of_range if index is an invalid column index
     */
    template <typename T_y, int R, int C>
    inline bool check_column_index(const std::string& function,
                                   const std::string& name,
                                   const Eigen::Matrix<T_y,R,C>& y,
                                   const size_t i) {
      if (i >= stan::error_index::value
          && i < static_cast<size_t>(y.cols()) + stan::error_index::value)
        return true;

      std::stringstream msg;
      msg << " for columns of " << name;
      out_of_range(function, 
                   y.cols(),
                   i, 
                   msg.str());
      return false;
    }

  }
}
#endif
