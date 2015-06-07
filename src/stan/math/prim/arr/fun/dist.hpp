#ifndef STAN_MATH_PRIM_ARR_FUN_DIST_HPP
#define STAN_MATH_PRIM_ARR_FUN_DIST_HPP

#include <vector>
#include <cstddef>
#include <cmath>

namespace stan {
  namespace math {

    inline double dist(const std::vector<double>& x,
                       const std::vector<double>& y) {
      using std::sqrt;
      double result = 0;
      for (size_t i = 0; i < x.size(); ++i) {
        double diff = x[i] - y[i];
        result += diff * diff;
      }
      return sqrt(result);
    }

  }
}

#endif
