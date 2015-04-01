#ifndef STAN__MATH__PRIM__MAT__META__GET_HPP
#define STAN__MATH__PRIM__MAT__META__GET_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {

  template <typename T, int R, int C>
  inline T get(const Eigen::Matrix<T, R, C>& m, size_t n) {
    return m(static_cast<int>(n));
  }

}
#endif

