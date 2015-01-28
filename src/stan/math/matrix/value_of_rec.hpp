#ifndef STAN__MATH__MATRIX__VALUE_OF_REC_HPP
#define STAN__MATH__MATRIX__VALUE_OF_REC_HPP

#include <stan/math/functions/value_of_rec.hpp>
#include <stan/math/meta/index_type.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    /**
     * Convert a matrix of type T to a matrix of doubles. 
     *
     * T must implement value_of_rec. See
     * test/agrad/fwd/matrix/value_of_test.cpp for fvar and var usage.
     *
     * @tparam T Scalar type in matrix
     * @tparam R Rows of matrix
     * @tparam C Columns of matrix
     * @param[in] M Matrix to be converted
     * @return Matrix of values 
     **/
    template<typename T, int R, int C>
    inline Eigen::Matrix<double,R,C> value_of_rec(const Eigen::Matrix<T,R,C>& M) {
      using stan::math::value_of_rec;
      Eigen::Matrix<double,R,C> Md(M.rows(),M.cols());
      for (int j = 0; j < M.cols(); j++)
        for (int i = 0; i < M.rows(); i++)

          Md(i,j) = value_of_rec(M(i,j));
      return Md;
    }
  }
}

#endif
