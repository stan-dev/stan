#ifndef STAN__MATH__MATRIX__VALUE_OF_HPP
#define STAN__MATH__MATRIX__VALUE_OF_HPP

#include <stan/math/functions/value_of.hpp>
#include <stan/math/meta/child_type.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    /**
     * Convert a matrix of type T to a matrix of doubles. 
     *
     * T must implement value_of. See
     * test/agrad/fwd/matrix/value_of.cpp for fvar and var usage.
     *
     * @tparam T Scalar type in matrix
     * @tparam R Rows of matrix
     * @tparam C Columns of matrix
     * @param[in] M Matrix to be converted
     * @return Matrix of values 
     **/
    template <typename T, int R, int C>
    inline Eigen::Matrix<typename child_type<T>::type,R,C> value_of(const Eigen::Matrix<T,R,C>& M) {
      using stan::math::value_of;
      Eigen::Matrix<typename child_type<T>::type,R,C> Md(M.rows(),M.cols());
      for (int j = 0; j < M.cols(); j++)
        for (int i = 0; i < M.rows(); i++)
          Md(i,j) = value_of(M(i,j));
      return Md;
    }
  }
}

#endif
