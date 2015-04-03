#ifndef STAN_MATH_PRIM_MAT_FUN_COLUMNS_DOT_PRODUCT_HPP
#define STAN_MATH_PRIM_MAT_FUN_COLUMNS_DOT_PRODUCT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/err/check_matching_sizes.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the dot product of the specified vectors.
     *
     * @param v1 First vector.
     * @param v2 Second vector.
     * @return Dot product of the vectors.
     * @throw std::domain_error If the vectors are not the same
     * size or if they are both not vector dimensioned.
     */
    template<int R1, int C1, int R2, int C2>
    inline Eigen::Matrix<double, 1, C1>
    columns_dot_product(const Eigen::Matrix<double, R1, C1>& v1,
                        const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::check_matching_sizes("columns_dot_product",
                                                 "v1", v1,
                                                 "v2", v2);
      Eigen::Matrix<double, 1, C1> ret(1, v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        ret(j) = v1.col(j).dot(v2.col(j));
      }
      return ret;
    }

  }
}
#endif
