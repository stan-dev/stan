#ifndef STAN__MATH__MATRIX__SUB_ROW_HPP
#define STAN__MATH__MATRIX__SUB_ROW_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_row_index.hpp>
#include <stan/math/error_handling/matrix/check_column_index.hpp>

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
      stan::math::check_row_index("sub_row(%1%)",i,m,"i",(double*)0);
      stan::math::check_column_index("sub_row(%1%)",j,m,"j",(double*)0);
      if (ncols > 0)
        stan::math::check_column_index("sub_col(%1%)",j+ncols-1,m,"j+ncols-1",
                                       (double*)0);
      return m.block(i - 1,j - 1,1,ncols);
    }

  }
}

#endif
