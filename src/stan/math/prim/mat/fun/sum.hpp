#ifndef STAN_MATH_PRIM_MAT_FUN_SUM_HPP
#define STAN_MATH_PRIM_MAT_FUN_SUM_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/arr/fun/sum.hpp>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Returns the sum of the coefficients of the specified
     * column vector.
     *
     * @tparam T Type of elements in matrix.
     * @tparam R Row type of matrix.
     * @tparam C Column type of matrix.
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
