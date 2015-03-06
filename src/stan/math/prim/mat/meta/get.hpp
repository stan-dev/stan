#ifndef STAN__MATH__PRIM__MAT__META__GET_HPP
#define STAN__MATH__PRIM__MAT__META__GET_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>

namespace stan {

  template <typename T>
  inline typename stan::math::value_type<T>::type get(const Eigen::MatrixBase<T>& m, size_t n) {
    return m(static_cast<int>(n));
  }

}
#endif

