#ifndef STAN_MATH_PRIM_SCAL_FUN_INC_BETA_DDA_HPP
#define STAN_MATH_PRIM_SCAL_FUN_INC_BETA_DDA_HPP

#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta_ddb.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T>
    T inc_beta_ddb(T a, T b, T z,
                   T digamma_b, T digamma_ab);

    /**
     * Returns the partial derivative of the incomplete beta function
     * with respect to a.
     *
     * @tparam T scalar types of arguments
     * @param a a
     * @param b b
     * @param z upper bound of the integral; must be greater than 0
     * @param digamma_a value of digamma(a)
     * @param digamma_ab value of digamma(b)
     * @return partial derivative of the incomplete beta with respect to a
     */
    template <typename T>
    T inc_beta_dda(T a, T b, T z,
                   T digamma_a, T digamma_ab) {
      using std::log;

      if (b > a)
        if ((0.1 < z && z <= 0.75 && b > 500)
            || (0.01 < z && z <= 0.1 && b > 2500)
            || (0.001 < z && z <= 0.01 && b > 1e5))
          return -inc_beta_ddb(b, a, 1 - z, digamma_a, digamma_ab);

      if (z > 0.75 && a < 500)
        return -inc_beta_ddb(b, a, 1 - z, digamma_a, digamma_ab);
      if (z > 0.9 && a < 2500)
        return -inc_beta_ddb(b, a, 1 - z, digamma_a, digamma_ab);
      if (z > 0.99 && a < 1e5)
        return -inc_beta_ddb(b, a, 1 - z, digamma_a, digamma_ab);
      if (z > 0.999)
        return -inc_beta_ddb(b, a, 1 - z, digamma_a, digamma_ab);

      double threshold = 1e-10;

      digamma_a += 1.0 / a;  // Need digamma(a + 1), not digamma(a);

      // Common prefactor to regularize numerator and denomentator
      T prefactor = (a + 1) / (a + b);
      prefactor = prefactor * prefactor * prefactor;

      T sum_numer = (digamma_ab - digamma_a) * prefactor;
      T sum_denom = prefactor;

      T summand = prefactor * z * (a + b) / (a + 1);

      T k = 1;
      digamma_ab += 1.0 / (a + b);
      digamma_a += 1.0 / (a + 1);

      while (fabs(summand) > threshold) {
        sum_numer += (digamma_ab - digamma_a) * summand;
        sum_denom += summand;

        summand *= (1 + (a + b) / k) * (1 + k) / (1 + (a + 1) / k);
        digamma_ab += 1.0 / (a + b + k);
        digamma_a += 1.0 / (a + 1 + k);
        ++k;
        summand *= z / k;

        if (k > 1e5)
          throw std::domain_error("stan::math::inc_beta_dda did "
                                  "not converge within 100000 iterations");
      }
      return inc_beta(a, b, z) * (log(z) + sum_numer / sum_denom);
    }

  }  // math
}   // stan

#endif
