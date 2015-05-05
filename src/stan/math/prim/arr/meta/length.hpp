#ifndef STAN_MATH_PRIM_ARR_META_LENGTH_HPP
#define STAN_MATH_PRIM_ARR_META_LENGTH_HPP

#include <cstdlib>
#include <vector>

namespace stan {

  template <typename T>
  size_t length(const std::vector<T>& x) {
    return x.size();
  }
}
#endif

