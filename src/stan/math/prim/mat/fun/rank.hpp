#ifndef STAN_MATH_PRIM_MAT_FUN_RANK_HPP
#define STAN_MATH_PRIM_MAT_FUN_RANK_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_range.hpp>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Return the number of components of v less than v[s].
     *
     * @return Number of components of v less than v[s].
     * @tparam T Type of elements of the vector.
     */
    template <typename T>
    inline size_t rank(const std::vector<T> & v, int s) {
      using stan::math::check_range;
      size_t size = v.size();
      check_range("rank", "v", size, s);
      s--;
      size_t count(0U);
      T compare(v[s]);
      for (size_t i = 0U; i < size; ++i)
        if (v[i] < compare)
          count++;
      return count;
    }

    /**
     * Return the number of components of v less than v[s].
     *
     * @return Number of components of v less than v[s].
     * @tparam T Type of elements of the vector.
     */
    template <typename T, int R, int C>
    inline size_t rank(const Eigen::Matrix<T, R, C> & v, int s) {
      using stan::math::check_range;
      size_t size = v.size();

      check_range("rank", "v", size, s);
      s--;
      const T * vv = v.data();
      size_t count(0U);
      T compare(vv[s]);
      for (size_t i = 0U; i < size; ++i)
        if (vv[i] < compare)
          count++;
      return count;
    }

  }
}
#endif
