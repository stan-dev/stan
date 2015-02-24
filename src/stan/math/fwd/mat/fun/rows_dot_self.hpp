#ifndef STAN__MATH__FWD__MAT__FUN__ROWS_DOT_SELF_HPP
#define STAN__MATH__FWD__MAT__FUN__ROWS_DOT_SELF_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/dot_self.hpp>

namespace stan {
  namespace agrad {

    template<typename T, int R,int C>
    inline Eigen::Matrix<fvar<T>,R,1> 
    rows_dot_self(const Eigen::Matrix<fvar<T>,R,C>& x) {
      Eigen::Matrix<fvar<T>,R,1> ret(x.rows(),1);
      for (size_type i = 0; i < x.rows(); i++) {
        Eigen::Matrix<fvar<T>,1,C> crow = x.row(i);
        ret(i,0) = dot_self(crow);
      }
      return ret;
    }
  }
}
#endif
