#ifndef STAN_MATH_PRIM_MAT_FUN_TO_ARRAY_2D_HPP
#define STAN_MATH_PRIM_MAT_FUN_TO_ARRAY_2D_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <vector>

namespace stan {
  namespace math {

    using Eigen::Dynamic;
    using Eigen::Matrix;
    using std::vector;

    // real[, ] to_array_2d(matrix)
    template <typename T>
    inline vector< vector<T> >
    to_array_2d(const Matrix<T, Dynamic, Dynamic> & matrix) {
      const T* datap = matrix.data();
      int C = matrix.cols();
      int R = matrix.rows();
      vector< vector<T> > result(R, vector<T>(C));
      for (int i=0, ij=0; i < C; i++)
        for (int j=0; j < R; j++, ij++)
          result[j][i] = datap[ij];
      return result;
    }

  }
}
#endif
