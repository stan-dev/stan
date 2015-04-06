#ifndef STAN_MATH_PRIM_MAT_META_LENGTH_HPP
#define STAN_MATH_PRIM_MAT_META_LENGTH_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {

  template <typename T, int R, int C>
  size_t length(const Eigen::Matrix<T, R, C>& m) {
    return m.size();
  }
}
#endif

