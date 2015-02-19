#ifndef STAN__MATH__PRIM__SCAL__META__LENGTH_HPP
#define STAN__MATH__PRIM__SCAL__META__LENGTH_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {


  // length() should only be applied to primitive or std vector or Eigen vector
  template <typename T>
  size_t length(const T& /*x*/) {
    return 1U;
  }
  template <typename T>
  size_t length(const std::vector<T>& x) {
    return x.size();
  }
  template <typename T, int R, int C>
  size_t length(const Eigen::Matrix<T,R,C>& m) {
    return m.size();
  }


}
#endif

