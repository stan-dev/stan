#ifndef STAN__MATH__PRIM__MAT__FUN__DOT_SELF_HPP
#define STAN__MATH__PRIM__MAT__FUN__DOT_SELF_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_vector.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the dot product of the specified vector with itself.
     * @param v Vector.
     * @tparam R number of rows or <code>Eigen::Dynamic</code> for dynamic
     * @tparam C number of rows or <code>Eigen::Dyanmic</code> for dynamic
     * @throw std::domain_error If v is not vector dimensioned.
     */
    template <int R, int C>
    inline double dot_self(const Eigen::Matrix<double, R, C>& v) {
      stan::math::check_vector("dot_self", "v", v);
      return v.squaredNorm();
    }

  }
}
#endif
