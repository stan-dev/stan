#ifndef STAN_MATH_FWD_MAT_FUN_SUM_HPP
#define STAN_MATH_FWD_MAT_FUN_SUM_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <vector>

namespace stan {
  namespace math {

    template <typename T, int R, int C>
    inline
    fvar<T>
    sum(const Eigen::Matrix<fvar<T>, R, C>& m) {
      fvar<T> sum = 0;
      if (m.size() == 0)
        return 0.0;
      for (unsigned i = 0; i < m.rows(); i++) {
        for (unsigned j = 0; j < m.cols(); j++)
          sum += m(i, j);
      }
      return sum;
    }
  }
}
#endif
