#ifndef STAN_MATH_PRIM_ARR_META_GET_HPP
#define STAN_MATH_PRIM_ARR_META_GET_HPP

#include <cstdlib>
#include <vector>

namespace stan {

  template <typename T>
  inline T get(const std::vector<T>& x, size_t n) {
    return x[n];
  }

}
#endif

