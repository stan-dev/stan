#ifndef STAN__MATH__MATRIX__APPEND__COL_HPP
#define STAN__MATH__MATRIX__APPEND__COL_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp> //stan::return_type
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <vector>

namespace stan {
  namespace math {
    
    using Eigen::Dynamic;
    using Eigen::Matrix; 
    using std::vector;
    using stan::error_handling::check_size_match;
       
    //matrix append_col(matrix, matrix)
    //matrix append_col(matrix, vector)
    //matrix append_col(vector, matrix)
    //matrix append_col(vector, vector)
    template <typename T1, typename T2, int R1, int C1, int R2, int C2>
    inline Matrix<typename return_type<T1, T2>::type, Dynamic, Dynamic>
    append_col(const Matrix<T1, R1, C1> & A,
          const Matrix<T2, R2, C2> & B) {
      int Arows = A.rows();
      int Brows = B.rows();
      int Acols = A.cols();
      int Bcols = B.cols();
      check_size_match("append_col",
                       "rows of A", Arows, 
                       "rows of B", Brows);
      
      Matrix<typename return_type<T1, T2>::type, Dynamic, Dynamic>
        result(Arows, Acols+Bcols);
      for (int j = 0; j < Acols; j++)
        for (int i = 0; i < Arows; i++)
          result(i, j) = A(i, j);
          
      for (int j = Acols, k = 0; k < Bcols; j++, k++)
        for (int i = 0; i < Arows; i++)
          result(i, j) = B(i, k);
      return result;
    }
       
    //row_vector append_col(row_vector, row_vector)
    template <typename T1, typename T2, int C1, int C2>
    inline Matrix<typename return_type<T1, T2>::type, 1, Dynamic>
    append_col(const Matrix<T1, 1, C1> & A,
          const Matrix<T2, 1, C2> & B) {
      int Asize = A.size();
      int Bsize = B.size();
      Matrix<typename return_type<T1, T2>::type, 1, Dynamic>
        result(Asize + Bsize);
      for (int i = 0; i < Asize; i++)
        result(i) = A(i);
      for (int i = 0, j = Asize; i < Bsize; i++, j++)
        result(j) = B(i);
      return result;
    }
    
       
    //matrix append_col(matrix, matrix)
    //matrix append_col(matrix, vector)
    //matrix append_col(vector, matrix)
    //matrix append_col(vector, vector)
    template <typename T, int R1, int C1, int R2, int C2>
    inline Matrix<T, Dynamic, Dynamic>
    append_col(const Matrix<T, R1, C1> & A,
          const Matrix<T, R2, C2> & B) {
      check_size_match("append_col",
                       "rows of A", A.rows(),
                       "rows of B", B.rows());
      
      Matrix<T, Dynamic, Dynamic>
        result(A.rows(), A.cols()+B.cols());
      result << A, B;
      return result;
    }
       
    //row_vector append_col(row_vector, row_vector)
    template <typename T, int C1, int C2>
    inline Matrix<T, 1, Dynamic>
    append_col(const Matrix<T, 1, C1> & A,
          const Matrix<T, 1, C2> & B) {          
      Matrix<T, 1, Dynamic>
        result(A.size()+B.size());
      result << A, B;
      return result;
    }
    
  }
}
#endif
