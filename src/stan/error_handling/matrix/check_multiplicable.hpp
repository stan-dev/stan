#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_MULTIPLICABLE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_MULTIPLICABLE_HPP

#include <sstream>
#include <stan/meta/traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/scalar/check_positive_index.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the matrices can be multiplied.
     *
     * This checks the runtime sizes to determine whether the two
     * matrices are multiplicable. This allows Eigen matrices,
     * vectors, and row vectors to be checked.
     *
     * @tparam T1 Type of first matrix
     * @tparam T2 Type of second matrix
     * 
     * @param function Function name (for error messages)
     * @param name1 Variable name for the first matrix (for error messages)
     * @param y1 First matrix
     * @param name2 Variable name for the second matrix (for error messages)
     * @param y2 Second matrix
     *
     * @return <code>true</code> if the two matrices are multiplicable
     * @throw <code>std::invalid_argument</code> if the matrices are not
     *   multiplicable or if either matrix is size 0 for either rows or columns
     */
    template <typename T1, typename T2>
    inline bool check_multiplicable(const std::string& function,
                                    const std::string& name1,
                                    const T1& y1,
                                    const std::string& name2,
                                    const T2& y2) {
      check_positive_index(function, name1, "rows()", y1.rows());
      check_positive_index(function, name2, "cols()", y2.cols());
      check_size_match(function, 
                       "Columns of matrix 1", y1.cols(), 
                       "Rows of matrix 2", y2.rows());
      check_positive_index(function, name1, "rows()", y1.cols());
      return true;
    }
  }
}
#endif
