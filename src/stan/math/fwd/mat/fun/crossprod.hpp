#ifndef STAN__MATH__FWD__MAT__FUN__CROSSPROD_HPP
#define STAN__MATH__FWD__MAT__FUN__CROSSPROD_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>

namespace stan {
  namespace agrad {

    template<typename T,int R, int C>
    inline
    Eigen::Matrix<fvar<T>,C,C>
    crossprod(const Eigen::Matrix<fvar<T>, R, C>& m) {
      if (m.rows() == 0)
        return Eigen::Matrix<fvar<T>,C,C>(0,0);
      return stan::agrad::multiply(stan::math::transpose(m), m);
    }

  }
}
#endif
