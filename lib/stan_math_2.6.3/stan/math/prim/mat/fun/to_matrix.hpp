#ifndef STAN_MATH_PRIM_MAT_FUN_TO_MATRIX_HPP
#define STAN_MATH_PRIM_MAT_FUN_TO_MATRIX_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
 // stan::scalar_type
#include <vector>

namespace stan {
  namespace math {

    using Eigen::Dynamic;
    using Eigen::Matrix;
    using std::vector;

    // matrix to_matrix(matrix)
    // matrix to_matrix(vector)
    // matrix to_matrix(row_vector)
    template <typename T, int R, int C>
    inline Matrix<T, Dynamic, Dynamic>
    to_matrix(Matrix<T, R, C> matrix) {
      return matrix;
    }

    // matrix to_matrix(real[, ])
    template <typename T>
    inline Matrix<T, Dynamic, Dynamic>
    to_matrix(const vector< vector<T> > & vec) {
      size_t R = vec.size();
      if (R != 0) {
        size_t C = vec[0].size();
        Matrix<T, Dynamic, Dynamic> result(R, C);
        T* datap = result.data();
        for (size_t i=0, ij=0; i < C; i++)
          for (size_t j=0; j < R; j++, ij++)
            datap[ij] = vec[j][i];
        return result;
      } else {
        return Matrix<T, Dynamic, Dynamic> (0, 0);
      }
    }

    // matrix to_matrix(int[, ])
    inline Matrix<double, Dynamic, Dynamic>
    to_matrix(const vector< vector<int> > & vec) {
      size_t R = vec.size();
      if (R != 0) {
        size_t C = vec[0].size();
        Matrix<double, Dynamic, Dynamic> result(R, C);
        double* datap = result.data();
        for (size_t i=0, ij=0; i < C; i++)
          for (size_t j=0; j < R; j++, ij++)
            datap[ij] = vec[j][i];
        return result;
      } else {
        return Matrix<double, Dynamic, Dynamic> (0, 0);
      }
    }

  }
}
#endif
