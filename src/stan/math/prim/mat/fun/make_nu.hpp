#ifndef STAN_MATH_PRIM_MAT_FUN_MAKE_NU_HPP
#define STAN_MATH_PRIM_MAT_FUN_MAKE_NU_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>

namespace stan {

  namespace math {

    /**
     * This function calculates the degrees of freedom for the t
     * distribution that corresponds to the shape parameter in the
     * Lewandowski et. al. distribution
     *
     * @param eta hyperparameter on (0, inf), eta = 1 <-> correlation
     * matrix is uniform
     * @param K number of variables in covariance matrix
     */
    template<typename T>
    const Eigen::Array<T, Eigen::Dynamic, 1>
    make_nu(const T eta, const size_t K) {
      using Eigen::Array;
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T, Dynamic, 1> >::type size_type;

      Array<T, Dynamic, 1> nu(K * (K - 1) / 2);

      T alpha = eta + (K - 2.0) / 2.0;  // from Lewandowski et. al.

      // Best (1978) implies nu = 2 * alpha for the dof in a t
      // distribution that generates a beta variate on (-1, 1)
      T alpha2 = 2.0 * alpha;
      for (size_type j = 0; j < (K - 1); j++) {
        nu(j) = alpha2;
      }
      size_t counter = K - 1;
      for (size_type i = 1; i < (K - 1); i++) {
        alpha -= 0.5;
        alpha2 = 2.0 * alpha;
        for (size_type j = i + 1; j < K; j++) {
          nu(counter) = alpha2;
          counter++;
        }
      }
      return nu;
    }

  }

}

#endif
