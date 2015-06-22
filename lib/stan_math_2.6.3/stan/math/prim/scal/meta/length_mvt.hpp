#ifndef STAN_MATH_PRIM_SCAL_META_LENGTH_MVT_HPP
#define STAN_MATH_PRIM_SCAL_META_LENGTH_MVT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stdexcept>
#include <vector>

namespace stan {

  // length_mvt() should only be applied to std vector or Eigen matrix
  template <typename T>
  size_t length_mvt(const T& ) {
    throw std::out_of_range("length_mvt passed to an unrecognized type.");
    return 1U;
  }
  template <typename T, int R, int C>
  size_t length_mvt(const Eigen::Matrix<T, R, C>& ) {
    return 1U;
  }
  template <typename T, int R, int C>
  size_t length_mvt(const std::vector<Eigen::Matrix<T, R, C> >& x) {
    return x.size();
  }

}
#endif

