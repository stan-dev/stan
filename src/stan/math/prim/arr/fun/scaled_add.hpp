#ifndef STAN_MATH_PRIM_ARR_FUN_SCALED_ADD_HPP
#define STAN_MATH_PRIM_ARR_FUN_SCALED_ADD_HPP

#include <vector>
#include <cstddef>

namespace stan {
  namespace math {

    // x <- x + lambda * y
    inline void scaled_add(std::vector<double>& x,
                           const std::vector<double>& y,
                           const double lambda) {
      for (size_t i = 0; i < x.size(); ++i)
        x[i] += lambda * y[i];
    }

  }
}

#endif
