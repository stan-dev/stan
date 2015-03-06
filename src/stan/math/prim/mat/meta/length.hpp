#ifndef STAN__MATH__PRIM__MAT__META__LENGTH_HPP
#define STAN__MATH__PRIM__MAT__META__LENGTH_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {

  template <typename T>
  size_t length(const Eigen::MatrixBase<T>& m) {
    return m.size();
  }
}
#endif

