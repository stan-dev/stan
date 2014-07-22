#ifndef STAN__MATH__MATRIX__BLOCK_HPP
#define STAN__MATH__MATRIX__BLOCK_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_row_index.hpp>
#include <stan/math/error_handling/matrix/check_column_index.hpp>

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
      check_row_index("block(%1%)",i,m,"i",(double*)0);
      check_row_index("block(%1%)",i+nrows-1,m,"i+nrows-1",(double*)0);
      check_column_index("block(%1%)",j,m,"j",(double*)0);
      check_column_index("block(%1%)",j+ncols-1,m,"j+ncols-1",(double*)0);
      return m.block(i - 1,j - 1,nrows,ncols);
    }
    
  }
}
#endif
