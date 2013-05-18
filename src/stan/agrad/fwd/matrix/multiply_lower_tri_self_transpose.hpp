#ifndef __STAN__AGRAD__FWD__MATRIX__MULTIPLY_LOWER_TRI_SELF_TRANSPOSE_HPP__
#define __STAN__AGRAD__FWD__MATRIX__MULTIPLY_LOWER_TRI_SELF_TRANSPOSE_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>

namespace stan {
  namespace agrad {
    
    template<typename T,int R, int C>
    inline
    Eigen::Matrix<fvar<T>,R,R> 
    multiply_lower_tri_self_transpose(const Eigen::Matrix<fvar<T>, R, C>& m) {
      if (m.rows() == 0)
        return Eigen::Matrix<fvar<T>,R,R>(0,0);
      Eigen::Matrix<fvar<T>,R,C> L(m.rows(),m.cols());
      L.setZero();

      for(size_type i = 0; i < m.rows(); i++) {
        for(size_type j = 0; (j < i + 1) && (j < m.cols()); j++)
          L(i,j) = m(i,j);
      }

      return stan::agrad::multiply(L, stan::math::transpose(L));
    }

  }
}
#endif
