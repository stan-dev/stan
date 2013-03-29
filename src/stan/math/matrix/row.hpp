#ifndef __STAN__MATH__MATRIX__ROW_HPP__
#define __STAN__MATH__MATRIX__ROW_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_row_index.hpp>

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
      validate_row_index(m,i,"row");
      return m.row(i - 1);
    }

  }
}
#endif
