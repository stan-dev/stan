#ifndef STAN_MATH_PRIM_ARR_FUN_SUM_HPP
#define STAN_MATH_PRIM_ARR_FUN_SUM_HPP

#include <vector>
#include <cstddef>

namespace stan {
  namespace math {

    inline double sum(std::vector<double>& x) {
      double sum = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        sum += x[i];
      return sum;
    }

  }
}

#endif
