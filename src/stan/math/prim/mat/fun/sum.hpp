#ifndef STAN__MATH__PRIM__MAT__FUN__SUM_HPP
#define STAN__MATH__PRIM__MAT__FUN__SUM_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Return the sum of the values in the specified
     * standard vector.
     *
     * @param xs Standard vector to sum.
     * @return Sum of elements.
     * @tparam T Type of elements summed.
     */
    template <typename T>
    inline T sum(const std::vector<T>& xs) {
      if (xs.size() == 0) return 0;
      T sum(xs[0]);
      for (size_t i = 1; i < xs.size(); ++i)
        sum += xs[i];
      return sum;
    }

    /**
     * Returns the sum of the coefficients of the specified
     * column vector.
     * @param v Specified vector.
     * @return Sum of coefficients of vector.
     */
    template <typename T, int R, int C>
    inline double sum(const Eigen::Matrix<T, R, C>& v) {
      return v.sum();
    }

  }
}
#endif
