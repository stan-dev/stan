#ifndef STAN_MATH_PRIM_MAT_FUN_UNIT_VECTOR_CONSTRAIN_HPP
#define STAN_MATH_PRIM_MAT_FUN_UNIT_VECTOR_CONSTRAIN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <cmath>

namespace stan {

  namespace math {

    // Unit vector

    /**
     * Return the unit length vector corresponding to the free vector y.
     * The free vector contains K-1 spherical coordinates.
     *
     * @param y of K - 1 spherical coordinates
     * @return Unit length vector of dimension K
     * @tparam T Scalar type.
     **/
    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1>
    unit_vector_constrain(const Eigen::Matrix<T, Eigen::Dynamic, 1>& y) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T, Dynamic, 1> >::type size_type;
      int Km1 = y.size();
      Matrix<T, Dynamic, 1> x(Km1 + 1);
      x(0) = 1.0;
      const T half_pi = T(M_PI/2.0);
      for (size_type k = 1; k <= Km1; ++k) {
        T yk_1 = y(k-1) + half_pi;
        T sin_yk_1 = sin(yk_1);
        x(k) = x(k-1)*sin_yk_1;
        x(k-1) *= cos(yk_1);
      }
      return x;
    }

    /**
     * Return the unit length vector corresponding to the free vector y.
     * The free vector contains K-1 spherical coordinates.
     *
     * @param y of K - 1 spherical coordinates
     * @return Unit length vector of dimension K
     * @param lp Log probability reference to increment.
     * @tparam T Scalar type.
     **/
    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1>
    unit_vector_constrain(const Eigen::Matrix<T, Eigen::Dynamic, 1>& y, T &lp) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T, Dynamic, 1> >::type size_type;

      int Km1 = y.size();
      Matrix<T, Dynamic, 1> x(Km1 + 1);
      x(0) = 1.0;
      const T half_pi = T(0.5 * M_PI);
      for (size_type k = 1; k <= Km1; ++k) {
        T yk_1 = y(k-1) + half_pi;
        T sin_yk_1 = sin(yk_1);
        x(k) = x(k-1) * sin_yk_1;
        x(k-1) *= cos(yk_1);
        if (k < Km1)
          lp += (Km1 - k) * log(fabs(sin_yk_1));
      }
      return x;
    }

  }

}

#endif
