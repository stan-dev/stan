#ifndef STAN__MATH__MATRIX__ROW_HPP
#define STAN__MATH__MATRIX__ROW_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/check_greater_or_equal.hpp>
#include <stan/error_handling/scalar/check_less_or_equal.hpp>

namespace stan {
  namespace math {

    /**
     * Return the specified row of the specified matrix, using
     * start-at-1 indexing.  
     *
     * This is equivalent to calling <code>m.row(i - 1)</code> and
     * assigning the resulting template expression to a row vector.
     * 
     * @tparam T Scalar value type for matrix.
     * @param m Matrix.
     * @param i Row index (count from 1).
     * @return Specified row of the matrix.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,1,Eigen::Dynamic>
    row(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m, 
        size_t i) {
      stan::error_handling::check_greater_or_equal("row", "i", i, 1U);
      stan::error_handling::check_less_or_equal("row", "i", i, static_cast<size_t>(m.rows()));     
      return m.row(i - 1);
    }

  }
}
#endif
