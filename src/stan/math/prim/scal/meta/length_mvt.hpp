#ifndef STAN__MATH__PRIM__SCAL__META__LENGTH_MVT_HPP
#define STAN__MATH__PRIM__SCAL__META__LENGTH_MVT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stdexcept>
#include <vector>

namespace stan {

  // length_mvt() should only be applied to std vector or Eigen matrix
  template <typename T>
  size_t length_mvt(const Eigen::MatrixBase<T>& ) {
    return 1U;
  }

  template <typename T>
  size_t length_mvt(const std::vector<T>& x) {
    return x.size();
  }

}
#endif

