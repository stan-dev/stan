#ifndef STAN__MATH__MATRIX__SUB_COL_HPP
#define STAN__MATH__MATRIX__SUB_COL_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_row_index.hpp>
#include <stan/error_handling/matrix/check_column_index.hpp>

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
      stan::error_handling::check_row_index("sub_col", "i", m, i);
      if (nrows > 0)
        stan::error_handling::check_row_index("sub_col", "i+nrows-1", m, i+nrows-1);
      stan::error_handling::check_column_index("sub_col", "j", m, j);
      return m.block(i - 1,j - 1,nrows,1);
    }
    

  }
}

#endif
