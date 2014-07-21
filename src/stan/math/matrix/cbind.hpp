#ifndef STAN__MATH__MATRIX__CBIND_HPP
#define STAN__MATH__MATRIX__CBIND_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp> //stan::return_type
#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <vector>

namespace stan {
  namespace math {
    
    using Eigen::Dynamic;
    using Eigen::Matrix; 
    using std::vector;
    using stan::math::check_size_match;
       
    //matrix cbind(matrix, matrix)
    //matrix cbind(matrix, vector)
    //matrix cbind(vector, matrix)
    //matrix cbind(vector, vector)
    template <typename T1, typename T2, int R1, int C1, int R2, int C2>
    inline Matrix<typename return_type<T1, T2>::type, Dynamic, Dynamic>
    cbind(const Matrix<T1, R1, C1> & A,
          const Matrix<T2, R2, C2> & B) {
      check_size_match("cbind(%1%)",
                       A.rows(), "rows of A",
                       B.rows(), "rows of B",
                       (double*)0);            
      
      Matrix<typename return_type<T1, T2>::type, Dynamic, Dynamic>
        result(A.rows(), A.cols()+B.cols());
      result << A, B;
      return result;
    }
       
    //matrix cbind(row_vector, row_vector)
    template <typename T1, typename T2, int C1, int C2>
    inline Matrix<typename return_type<T1, T2>::type, 1, Dynamic>
    cbind(const Matrix<T1, 1, C1> & A,
          const Matrix<T2, 1, C2> & B) {          
      Matrix<typename return_type<T1, T2>::type, 1, Dynamic>
        result(A.size()+B.size());
      result << A, B;
      return result;
    }
    
  }
}
#endif
