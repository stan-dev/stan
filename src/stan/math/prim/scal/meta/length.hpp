#ifndef STAN__MATH__PRIM__SCAL__META__LENGTH_HPP
#define STAN__MATH__PRIM__SCAL__META__LENGTH_HPP

#include <cstdlib>

namespace stan {

  template <typename T>
  size_t length(const T& /*x*/) {
    return 1U;
  }
}
#endif

