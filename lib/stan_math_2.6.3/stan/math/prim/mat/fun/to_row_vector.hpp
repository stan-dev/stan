#ifndef STAN_MATH_PRIM_MAT_FUN_TO_ROW_VECTOR_HPP
#define STAN_MATH_PRIM_MAT_FUN_TO_ROW_VECTOR_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
 // stan::scalar_type
#include <vector>

namespace stan {
  namespace math {

    using Eigen::Dynamic;
    using Eigen::Matrix;
    using std::vector;

    // row_vector to_row_vector(matrix)
    // row_vector to_row_vector(vector)
    // row_vector to_row_vector(row_vector)
    template <typename T, int R, int C>
    inline Matrix<T, 1, Dynamic>
    to_row_vector(const Matrix<T, R, C>& matrix) {
      return Matrix<T, 1, Dynamic>::Map(matrix.data(),
                                        matrix.rows()*matrix.cols());
    }

    // row_vector to_row_vector(real[])
    template <typename T>
    inline Matrix<T, 1, Dynamic>
    to_row_vector(const vector<T> & vec) {
      return Matrix<T, 1, Dynamic>::Map(vec.data(), vec.size());
    }

    // row_vector to_row_vector(int[])
    inline Matrix<double, 1, Dynamic>
    to_row_vector(const vector<int> & vec) {
      int C = vec.size();
      Matrix<double, 1, Dynamic> result(C);
      double* datap = result.data();
      for (int i=0; i < C; i++)
        datap[i] = vec[i];
      return result;
    }

  }
}
#endif
