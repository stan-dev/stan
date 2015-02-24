#ifndef STAN__MATH__PRIM__ARR__META__LENGTH_HPP
#define STAN__MATH__PRIM__ARR__META__LENGTH_HPP

#include <vector>

namespace stan {

  template <typename T>
  size_t length(const std::vector<T>& x) {
    return x.size();
  }
}
#endif

