#ifndef STAN__MATH__PRIM__MAT__FUN__RANK_HPP
#define STAN__MATH__PRIM__MAT__FUN__RANK_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_range.hpp>

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
        if (v[i]<compare) count++;
      return count;
    }

    /**
     * Return the number of components of v less than v[s].
     *
     * @return Number of components of v less than v[s].
     * @tparam T Type of elements of the vector.
     */
    template <typename T, int R, int C>
    inline size_t rank(const Eigen::Matrix<T,R,C> & v, int s) {
      using stan::math::check_range;
      size_t size = v.size();

      check_range("rank", "v", size, s);
      s--;
      const T * vv = v.data();
      size_t count(0U);
      T compare(vv[s]);
      for (size_t i = 0U; i < size; ++i)
        if (vv[i]<compare) count++;
      return count;
    }

  }
}
#endif
