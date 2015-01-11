#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_COLUMN_INDEX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_COLUMN_INDEX_HPP

#include <sstream>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified index is a valid column of the matrix
     *
     * NOTE: this will not throw if y contains nan values.
     *
     * @param function
     * @param i is index
     * @param y Matrix to test against
     * @param name
     * @return <code>true</code> if the index is a valid column index of the matrix.
     * @tparam T Type of scalar.
     */
    template <typename T_y, int R, int C>
    inline bool check_column_index(const std::string& function,
                                   const std::string& name,
                                   const Eigen::Matrix<T_y,R,C>& y,
                                   size_t i) {
      if ((i > 0) && (i <= static_cast<size_t>(y.cols())))
        return true;

      std::ostringstream msg;
      msg << ") must be greater than 0 and less than " 
          << y.cols();
      dom_err(function, name, i,
              "(", msg.str());
      return false;
    }

  }
}
#endif
