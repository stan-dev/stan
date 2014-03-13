#ifndef __STAN__MATH__MATRIX__CONTAINERS_CONVERSION_HPP__
#define __STAN__MATH__MATRIX__CONTAINERS_CONVERSION_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/dims.hpp>
#include <stan/meta/traits.hpp> //stan::scalar_type
#include <vector>

namespace stan {
  namespace math {
    
    using Eigen::Dynamic;
    using Eigen::Matrix; 
    using std::vector;
    
    //matrix to_matrix(row_vector)
    template <typename T>
    inline Matrix<T, Dynamic, Dynamic>
    to_matrix(Matrix<T, 1, Dynamic> matrix) {
      return matrix;
    }

    //matrix to_matrix(vector)    
    template <typename T>
    inline Matrix<T, Dynamic, Dynamic>
    to_matrix(Matrix<T, Dynamic, 1> matrix) {
      return matrix;
    }

    //matrix to_matrix(real[,])
    template <typename T>
    inline Matrix<T, Dynamic, Dynamic>
    to_matrix(const std::vector< std::vector<T> > & vec) {
      vector<int> RC = dims(vec);
      int R = RC[0];
      int C = RC[1];
      Matrix<T, Dynamic, Dynamic> result(R, C);
      T* datap = result.data();
      for (int i=0, ij=0; i < C; i++)
        for (int j=0; j < R; j++, ij++)
          datap[ij] = vec[j][i];
      return result;
    }
    
    //matrix to_matrix(int[,])
    inline Matrix<double, Dynamic, Dynamic>
    to_matrix(const std::vector< std::vector<int> > & vec) {
      vector<int> RC = dims(vec);
      int R = RC[0];
      int C = RC[1];
      Matrix<double, Dynamic, Dynamic> result(R, C);
      double* datap = result.data();
      for (int i=0, ij=0; i < C; i++)
        for (int j=0; j < R; j++, ij++)
          datap[ij] = vec[j][i];
      return result;
    }

    //vector to_vector(matrix)
    template <typename T, int R, int C>
    inline Matrix<T, Dynamic, 1>
    to_vector(const Matrix<T, R, C>& matrix) {
      return Matrix<T, Dynamic, 1>::Map(matrix.data(), matrix.rows()*matrix.cols());
    }

    //vector to_vector(row_vector)
    template <typename T>
    inline Matrix<T, Dynamic, 1>
    to_vector(Matrix<T, 1, Dynamic> vec) {
      return vec;
    }

    //row_vector to_row_vector(matrix)
    template <typename T, int R, int C>
    inline Matrix<T, 1, Dynamic>
    to_row_vector(const Matrix<T, R, C>& matrix) {
      return Matrix<T, 1, Dynamic>::Map(matrix.data(), matrix.rows()*matrix.cols());
    }

    //row_vector to_row_vector(vector)
    template <typename T>
    inline Matrix<T, 1, Dynamic>
    to_row_vector(Matrix<T, Dynamic, 1> vec) {
      return vec;
    }
    
    //vector to_vector(real[])
    template <typename T>
    inline Matrix<T, Dynamic, 1>
    to_vector(const std::vector<T> & vec) {
      return Matrix<T, Dynamic, 1>::Map(vec.data(), vec.size());
    }
    
    //vector to_vector(int[])
    inline Matrix<double, Dynamic, 1>
    to_vector(const std::vector<int> & vec) {
      int R = vec.size();
      Matrix<double, Dynamic, 1> result(R);
      double* datap = result.data();
      for (int i=0; i < R; i++)
        datap[i] = vec[i];
      return result;
    }
    
    //row_vector to_row_vector(real[])
    template <typename T>
    inline Matrix<T, 1, Dynamic>
    to_row_vector(const std::vector<T> & vec) {
      return Matrix<T, 1, Dynamic>::Map(vec.data(), vec.size());
    }
    
    //row_vector to_row_vector(int[])
    inline Matrix<double, 1, Dynamic>
    to_row_vector(const std::vector<int> & vec) {
      int C = vec.size();
      Matrix<double, 1, Dynamic> result(C);
      double* datap = result.data();
      for (int i=0; i < C; i++)
        datap[i] = vec[i];
      return result;
    }
    
    //real[,] to_array(matrix)
    template <typename T>
    inline vector< vector<T> >
    to_array(const Matrix<T, Dynamic, Dynamic> & matrix) {
      const T* datap = matrix.data();
      int C = matrix.cols();
      int R = matrix.rows();
      vector< vector<T> > result(R, vector<T>(C));
      for (int i=0, ij=0; i < C; i++)
        for (int j=0; j < R; j++, ij++)
          result[j][i] = datap[ij];
      return result;
    }

    //real[] to_array(row_vector)
    template <typename T>
    inline vector<T> to_array(const Matrix<T, 1, Dynamic> & vec) {
      const T* datap = vec.data();
      int C = vec.cols();
      vector<T> result(C);
      for (int i=0; i < C; i++)
        result[i] = datap[i];
      return result;
    }

    //real[] to_array(vector)
    template <typename T>
    inline vector<T>
    to_array(const Matrix<T, Dynamic, 1> & vec) {
      const T* datap = vec.data();
      int R = vec.rows();
      vector<T> result(R);
      for (int i=0; i < R; i++)
        result[i] = datap[i];
      return result;
    }
    
    template <typename T>
    inline std::vector<T>
    flatten(const std::vector<T> & x) {
      return x;
    }
    
    template <typename T>
    inline std::vector<typename stan::scalar_type<T>::type>
    flatten(const std::vector< std::vector<T> > & x) {
      size_t size1 = x.size();
      size_t size2 = 0;
      if (size1 != 0)
        size2 = x[0].size();
      std::vector<T> y(size1*size2);
      for(size_t i=0, ij=0; i < size1; i++)
        for(size_t j=0; j < size2; j++, ij++)
          y[ij] = x[i][j];
      return flatten(y);
    }
    
  }
}
#endif
