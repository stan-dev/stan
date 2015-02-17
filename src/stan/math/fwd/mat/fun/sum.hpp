#ifndef STAN__MATH__FWD__MAT__FUN__SUM_HPP
#define STAN__MATH__FWD__MAT__FUN__SUM_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core/fvar.hpp>

namespace stan {
  namespace agrad {

    template <typename T, int R, int C>
    inline 
    fvar<T> 
    sum(const Eigen::Matrix<fvar<T>,R,C>& m) {
      fvar<T> sum = 0;
      if (m.size() == 0)
        return 0.0;
      for(unsigned i = 0; i < m.rows(); i++) {
        for(unsigned j = 0; j < m.cols(); j++)
          sum += m(i,j);
      }
      return sum;
    }
  }
}
#endif
