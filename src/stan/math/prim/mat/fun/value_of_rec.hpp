#ifndef STAN__MATH__PRIM__MAT__FUN__VALUE_OF_REC_HPP
#define STAN__MATH__PRIM__MAT__FUN__VALUE_OF_REC_HPP

#include <stan/math/prim/scal/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace math {
    
    /**
     * Convert a matrix of type T to a matrix of doubles. 
     *
     * The scalar type of T must implement value_of_rec. See
     * test/agrad/fwd/matrix/value_of_test.cpp for fvar and var usage.
     *
     * @tparam T the type of the matrix
     * @param[in] M Matrix to be converted
     * @return Matrix of values 
     **/
    template <typename T>
    inline Eigen::MatrixXd value_of_rec(const Eigen::MatrixBase<T>& M) {
      using stan::math::value_of_rec;
      Eigen::MatrixXd Md(M.rows(),M.cols());
      for (int j = 0; j < M.cols(); j++)
        for (int i = 0; i < M.rows(); i++)
          Md(i,j) = value_of_rec(M(i,j));
      return Md;
    }
  }
}

#endif
