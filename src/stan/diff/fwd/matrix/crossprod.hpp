#ifndef __STAN__DIFF__FWD__MATRIX__CROSSPROD_HPP__
#define __STAN__DIFF__FWD__MATRIX__CROSSPROD_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/diff/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>

namespace stan {
  namespace diff {
    
    template<typename T,int R, int C>
    inline
    Eigen::Matrix<fvar<T>,C,C> 
    crossprod(const Eigen::Matrix<fvar<T>, R, C>& m) {
      if (m.rows() == 0)
        return Eigen::Matrix<fvar<T>,C,C>(0,0);
      return stan::diff::multiply(stan::math::transpose(m), m);
    }

  }
}
#endif
