#ifndef STAN_MATH_PRIM_ARR_FUN_SUM_HPP
#define STAN_MATH_PRIM_ARR_FUN_SUM_HPP

#include <cstddef>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Return the sum of the values in the specified standard vector.
     *
     * @tparam T Type of elements summed.
     * @param xs Standard vector to sum.
     * @return Sum of elements.
     */
    template <typename T>
    inline T sum(const std::vector<T>& xs) {
      if (xs.size() == 0) return 0;
      T sum(xs[0]);
      for (size_t i = 1; i < xs.size(); ++i)
        sum += xs[i];
      return sum;
    }

  }
}
#endif
