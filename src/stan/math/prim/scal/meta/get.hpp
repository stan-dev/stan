#ifndef STAN__MATH__PRIM__SCAL__META__GET_HPP
#define STAN__MATH__PRIM__SCAL__META__GET_HPP

#include <cmath>
#include <cstddef>

namespace stan {

  template <typename T>
  inline T get(const T& x, size_t n) {
    return x;
  }

}
#endif

