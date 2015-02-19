#ifndef STAN__MATH__PRIM__SCAL__META__GET_HPP
#define STAN__MATH__PRIM__SCAL__META__GET_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {

  template <typename T>
  inline T get(const T& x, size_t n) {
    return x;
  }
  template <typename T>
  inline T get(const std::vector<T>& x, size_t n) {
    return x[n];
  }
  template <typename T, int R, int C>
  inline T get(const Eigen::Matrix<T,R,C>& m, size_t n) {
    return m(static_cast<int>(n));
  }

}
#endif

