#ifndef __STAN__MATH__MATRIX__BLOCK_HPP__
#define __STAN__MATH__MATRIX__BLOCK_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_row_index.hpp>
#include <stan/math/matrix/validate_column_index.hpp>

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
      validate_row_index(m,i,"block");
      validate_row_index(m,i+nrows-1,"block");
      validate_column_index(m,j,"block");
      validate_column_index(m,j+ncols-1,"block");
      return m.block(i - 1,j - 1,nrows,ncols);
    }



    
    
  }
}
#endif
