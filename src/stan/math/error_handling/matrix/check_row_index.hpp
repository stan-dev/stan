#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_ROW_INDEX_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_ROW_INDEX_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/dom_err.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified index is a valid row of the matrix
     *
     * @param function
     * @param i is index
     * @param y Matrix to test against
     * @param name
     * @param result
     * @return <code>true</code> if the index is a valid row index of the matrix.
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result, int R, int C>
    inline bool check_row_index(const char* function,
                                size_t i,
                                const Eigen::Matrix<T_y,R,C>& y,
                                const char* name,
                                T_result* result) {
      if ((i > 0) && (i <= static_cast<size_t>(y.rows())))
        return true;

      std::ostringstream msg;
      msg << name << " (%1%) must be greater than 0 and less than " 
          << y.rows();
      std::string tmp(msg.str());
      return dom_err(function,i,name,
                     tmp.c_str(),"",
                     result);
      
    }

  }
}
#endif
