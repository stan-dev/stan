#ifndef STAN__MATH__MATRIX__CONTAINERS_CONVERSION_HPP
#define STAN__MATH__MATRIX__CONTAINERS_CONVERSION_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp> //stan::scalar_type
#include <vector>

namespace stan {
  namespace math {
    
    using Eigen::Dynamic;
    using Eigen::Matrix; 
    using std::vector;
    
    //matrix to_matrix(matrix)    
    //matrix to_matrix(vector)    
    //matrix to_matrix(row_vector)
    template <typename T, int R, int C>
    inline Matrix<T, Dynamic, Dynamic>
    to_matrix(Matrix<T, R, C> matrix) {
      return matrix;
    }

    //matrix to_matrix(real[,])
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
    
    //matrix to_matrix(int[,])
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

    //vector to_vector(matrix)
    //vector to_vector(row_vector)
    //vector to_vector(vector)
    template <typename T, int R, int C>
    inline Matrix<T, Dynamic, 1>
    to_vector(const Matrix<T, R, C>& matrix) {
      return Matrix<T, Dynamic, 1>::Map(matrix.data(), matrix.rows()*matrix.cols());
    }
    
    //vector to_vector(real[])
    template <typename T>
    inline Matrix<T, Dynamic, 1>
    to_vector(const vector<T> & vec) {
      return Matrix<T, Dynamic, 1>::Map(vec.data(), vec.size());
    }
    
    //vector to_vector(int[])
    inline Matrix<double, Dynamic, 1>
    to_vector(const vector<int> & vec) {
      int R = vec.size();
      Matrix<double, Dynamic, 1> result(R);
      double* datap = result.data();
      for (int i=0; i < R; i++)
        datap[i] = vec[i];
      return result;
    }
    
    //row_vector to_row_vector(matrix)
    //row_vector to_row_vector(vector)
    //row_vector to_row_vector(row_vector)
    template <typename T, int R, int C>
    inline Matrix<T, 1, Dynamic>
    to_row_vector(const Matrix<T, R, C>& matrix) {
      return Matrix<T, 1, Dynamic>::Map(matrix.data(), matrix.rows()*matrix.cols());
    }  
      
    //row_vector to_row_vector(real[])
    template <typename T>
    inline Matrix<T, 1, Dynamic>
    to_row_vector(const vector<T> & vec) {
      return Matrix<T, 1, Dynamic>::Map(vec.data(), vec.size());
    }
    
    //row_vector to_row_vector(int[])
    inline Matrix<double, 1, Dynamic>
    to_row_vector(const vector<int> & vec) {
      int C = vec.size();
      Matrix<double, 1, Dynamic> result(C);
      double* datap = result.data();
      for (int i=0; i < C; i++)
        datap[i] = vec[i];
      return result;
    }
    
    //real[,] to_array_2d(matrix)
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
