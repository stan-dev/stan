#ifndef STAN__MATH__PRIM__MAT__FUN__TO_ARRAY_1D_HPP
#define STAN__MATH__PRIM__MAT__FUN__TO_ARRAY_1D_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <vector>

namespace stan {
  namespace math {

    using Eigen::Dynamic;
    using Eigen::Matrix;
    using std::vector;

    //real[] to_array_1d(matrix)
    //real[] to_array_1d(row_vector)
    //real[] to_array_1d(vector)
    template <typename T, int R, int C>
    inline vector<T> to_array_1d(const Matrix<T, R, C> & matrix) {
      const T* datap = matrix.data();
      int size = matrix.size();
      vector<T> result(size);
      for (int i=0; i < size; i++)
        result[i] = datap[i];
      return result;
    }

    //real[] to_array_1d(...)
    template <typename T>
    inline vector<T>
    to_array_1d(const vector<T> & x) {
      return x;
    }

    //real[] to_array_1d(...)
    template <typename T>
    inline vector<typename scalar_type<T>::type>
    to_array_1d(const vector< vector<T> > & x) {
      size_t size1 = x.size();
      size_t size2 = 0;
      if (size1 != 0)
        size2 = x[0].size();
      vector<T> y(size1*size2);
      for(size_t i=0, ij=0; i < size1; i++)
        for(size_t j=0; j < size2; j++, ij++)
          y[ij] = x[i][j];
      return to_array_1d(y);
    }

  }
}
#endif
