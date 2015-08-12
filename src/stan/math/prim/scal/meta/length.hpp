#ifndef STAN_MATH_PRIM_SCAL_META_LENGTH_HPP
#define STAN_MATH_PRIM_SCAL_META_LENGTH_HPP

#include <cstdlib>

namespace stan {

  template <typename T>
  size_t length(const T& /*x*/) {
    return 1U;
  }
}
#endif

