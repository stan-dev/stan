#ifndef STAN__MATH__PRIM__MAT__FUN__BLOCK_HPP
#define STAN__MATH__PRIM__MAT__FUN__BLOCK_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_row_index.hpp>
#include <stan/math/prim/mat/err/check_column_index.hpp>

namespace stan {
  namespace math {

    /**
     * Return a nrows x ncols submatrix starting at (i-1,j-1).
     *
     * @param m Matrix
     * @param i Starting row
     * @param j Starting column
     * @param nrows Number of rows in block
     * @param ncols Number of columns in block
     **/
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    block(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m,
          size_t i, size_t j, size_t nrows, size_t ncols) {
      using stan::math::check_row_index;
      using stan::math::check_column_index;

      check_row_index("block", "i", m, i);
      check_row_index("block", "i+nrows-1", m, i+nrows-1);
      check_column_index("block", "j", m, j);
      check_column_index("block", "j+ncols-1", m, j+ncols-1);
      return m.block(i - 1,j - 1,nrows,ncols);
    }

  }
}
#endif
