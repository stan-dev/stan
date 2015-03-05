#ifndef STAN__MATH__PRIM__ARR__META__GET_HPP
#define STAN__MATH__PRIM__ARR__META__GET_HPP

#include <cstdlib>
#include <vector>

namespace stan {

  template <typename T>
  inline T get(const std::vector<T>& x, size_t n) {
    return x[n];
  }

}
#endif

