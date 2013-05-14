#ifndef __STAN__MATH__MATRIX__SUB_COL_HPP__
#define __STAN__MATH__MATRIX__SUB_COL_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_row_index.hpp>
#include <stan/math/matrix/validate_column_index.hpp>

namespace stan {
  namespace math {

    /**
     * Return a nrows x 1 subcolumn starting at (i-1,j-1).
     *
     * @param m Matrix
     * @param i Starting row + 1
     * @param j Starting column + 1
     * @param nrows Number of rows in block
     **/
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    sub_col(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m,
          size_t i, size_t j, size_t nrows) {
      validate_row_index(m,i,"sub_column");
      if (nrows > 0) validate_row_index(m,i+nrows-1,"sub_column");
      validate_column_index(m,j,"sub_column");
      return m.block(i - 1,j - 1,nrows,1);
    }
    

  }
}

#endif
