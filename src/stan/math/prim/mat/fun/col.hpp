#ifndef STAN_MATH_PRIM_MAT_FUN_COL_HPP
#define STAN_MATH_PRIM_MAT_FUN_COL_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_column_index.hpp>

namespace stan {
  namespace math {

    /**
     * Return the specified column of the specified matrix
     * using start-at-1 indexing.
     *
     * This is equivalent to calling <code>m.col(i - 1)</code> and
     * assigning the resulting template expression to a column vector.
     *
     * @param m Matrix.
     * @param j Column index (count from 1).
     * @return Specified column of the matrix.
     */
    template <typename T>
    inline
    Eigen::Matrix<T, Eigen::Dynamic, 1>
    col(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m,
        size_t j) {
      stan::math::check_column_index("col", "j", m, j);
      return m.col(j - 1);
    }

  }
}
#endif
