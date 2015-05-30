#ifndef STAN_MATH_REV_MAT_FUN_SORT_ASC_HPP
#define STAN_MATH_REV_MAT_FUN_SORT_ASC_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <algorithm>    // std::sort
#include <functional>   // std::greater
#include <valarray>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Return the specified standard vector in ascending order with gradients kept.
     *
     * @param xs Standard vector to order.
     * @return Standard vector ordered.
     * @tparam T Type of elements of the vector.
     */
    inline std::vector<var> sort_asc(std::vector<var> xs) {
      std::sort(xs.begin(), xs.end());
      return xs;
    }

    /**
     * Return the specified eigen vector in ascending order with gradients kept.
     *
     * @param xs Eigen vector to order.
     * @return Eigen vector ordered.
     * @tparam T Type of elements of the vector.
     */
    template <int R, int C>
    inline typename Eigen::Matrix<var, R, C>
    sort_asc(Eigen::Matrix<var, R, C> xs) {
      std::sort(xs.data(), xs.data()+xs.size());
      return xs;
    }

  }
}
#endif
