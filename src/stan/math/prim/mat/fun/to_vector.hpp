#ifndef STAN_MATH_PRIM_MAT_FUN_TO_VECTOR_HPP
#define STAN_MATH_PRIM_MAT_FUN_TO_VECTOR_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
 // stan::scalar_type
#include <vector>

namespace stan {
  namespace math {

    using Eigen::Dynamic;
    using Eigen::Matrix;
    using std::vector;

    // vector to_vector(matrix)
    // vector to_vector(row_vector)
    // vector to_vector(vector)
    template <typename T, int R, int C>
    inline Matrix<T, Dynamic, 1>
    to_vector(const Matrix<T, R, C>& matrix) {
      return Matrix<T, Dynamic, 1>::Map(matrix.data(),
                                        matrix.rows()*matrix.cols());
    }

    // vector to_vector(real[])
    template <typename T>
    inline Matrix<T, Dynamic, 1>
    to_vector(const vector<T> & vec) {
      return Matrix<T, Dynamic, 1>::Map(vec.data(), vec.size());
    }

    // vector to_vector(int[])
    inline Matrix<double, Dynamic, 1>
    to_vector(const vector<int> & vec) {
      int R = vec.size();
      Matrix<double, Dynamic, 1> result(R);
      double* datap = result.data();
      for (int i=0; i < R; i++)
        datap[i] = vec[i];
      return result;
    }


  }
}
#endif
