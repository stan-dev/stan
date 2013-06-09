#ifndef __STAN__MATH__MATRIX__SUB_ROW_HPP__
#define __STAN__MATH__MATRIX__SUB_ROW_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_row_index.hpp>
#include <stan/math/matrix/validate_column_index.hpp>

namespace stan {

  namespace math {

    /**
     * Return a 1 x nrows subrow starting at (i-1,j-1).
     *
     * @param m Matrix
     * @param i Starting row + 1
     * @param j Starting column + 1
     * @param ncols Number of columns in block
     **/
    template <typename T>
    inline
    Eigen::Matrix<T,1,Eigen::Dynamic>
    sub_row(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m,
               size_t i, size_t j, size_t ncols) {
      validate_row_index(m,i,"sub_row");
      validate_column_index(m,j,"sub_row");
      if (ncols > 0) validate_column_index(m,j+ncols-1,"sub_row");
      return m.block(i - 1,j - 1,1,ncols);
    }

  }
}

#endif
