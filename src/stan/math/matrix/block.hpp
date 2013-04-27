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
      validate_row_index(m,i+nrows-1,"sub_column");
      validate_column_index(m,j,"sub_column");
      return m.block(i - 1,j - 1,nrows,1);
    }
    
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
      validate_column_index(m,j+ncols-1,"sub_row");
      return m.block(i - 1,j - 1,1,ncols);
    }

    /**
     * Return the specified number of elements as a vector
     * from the front of the specified vector.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    head(const Eigen::Matrix<T,Eigen::Dynamic,1>& v,
         size_t n) {
      validate_row_index(v, v.size() - n, "head");
      return v.head(n);
    }

    /**
     * Return the specified number of elements as a vector
     * from the back of the specified vector.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    tail(const Eigen::Matrix<T,Eigen::Dynamic,1>& v,
         size_t n) {
      validate_row_index(v, v.size() - n, "tail");
      return v.tail(n);
    }

    /**
     * Return the specified number of elements as a vector starting
     * from the specified element - 1 of the specified vector.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    segment(const Eigen::Matrix<T,Eigen::Dynamic,1>& v,
            size_t i, size_t n) {
      validate_row_index(v,i,"segment");
      validate_row_index(v, i + n - 1, "segment");
      return v.segiment(i-1,n);
    }

    
    
  }
}
#endif
