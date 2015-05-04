#ifndef STAN_MATH_PRIM_SCAL_FUN_INC_BETA_DERIVATIVES_HPP
#define STAN_MATH_PRIM_SCAL_FUN_INC_BETA_DERIVATIVES_HPP

#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <cmath>

namespace stan {
  namespace math {

    // Gradients of the regularized incomplete beta function ibeta(a, b, z)

    template <typename T>
    T ddz_inc_beta(T a, T b, T z) {
      return exp((b - 1) * log(1 - z) + (a - 1) * log(z)
                 + lgamma(a + b) - lgamma(a) - lgamma(b));
    }

    template <>
    double ddz_inc_beta(double a, double b, double z) {
      using boost::math::ibeta_derivative;
      return ibeta_derivative(a, b, z);
    }

    template <typename T>
    T dda_inc_beta(T a, T b, T z,
                   T digamma_a, T digamma_ab);

    template <typename T>
    T ddb_inc_beta(T a, T b, T z,
                   T digamma_b, T digamma_ab);

    template <typename T>
    T dda_inc_beta(T a, T b, T z,
                   T digamma_a, T digamma_ab) {
      using std::log;

      if (z > 0.5 && a < 250)
        return -ddb_inc_beta(b, a, 1 - z, digamma_a, digamma_ab);
      if (z > 0.75 && a < 500)
        return -ddb_inc_beta(b, a, 1 - z, digamma_a, digamma_ab);
      if (z > 0.9 && a < 2500)
        return -ddb_inc_beta(b, a, 1 - z, digamma_a, digamma_ab);
      if (z > 0.99 && a < 1e5)
        return -ddb_inc_beta(b, a, 1 - z, digamma_a, digamma_ab);
      if (z > 0.999)
        return -ddb_inc_beta(b, a, 1 - z, digamma_a, digamma_ab);

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
          throw std::domain_error("stan::math::dda_inc_beta did "
                                  "not converge within 100000 iterations");
      }
      return inc_beta(a, b, z) * (log(z) + sum_numer / sum_denom);
    }

    template <typename T>
    T ddb_inc_beta(T a, T b, T z,
                   T digamma_b, T digamma_ab) {
      using std::log;

      if (z > 0.5 && a < 250)
        return -dda_inc_beta(b, a, 1 - z, digamma_b, digamma_ab);
      if (z > 0.75 && a < 500)
        return -dda_inc_beta(b, a, 1 - z, digamma_b, digamma_ab);
      if (z > 0.9 && a < 2500)
        return -dda_inc_beta(b, a, 1 - z, digamma_b, digamma_ab);
      if (z > 0.99 && a < 1e5)
        return -dda_inc_beta(b, a, 1 - z, digamma_b, digamma_ab);
      if (z > 0.999)
        return -dda_inc_beta(b, a, 1 - z, digamma_b, digamma_ab);

      double threshold = 1e-10;

      // Common prefactor to regularize numerator and denomentator
      T prefactor = (a + 1) / (a + b);
      prefactor = prefactor * prefactor * prefactor;

      T sum_numer = digamma_ab * prefactor;
      T sum_denom = prefactor;

      T summand = prefactor * z * (a + b) / (a + 1);

      T k = 1;
      digamma_ab += 1.0 / (a + b);

      while (fabs(summand) > threshold) {
        sum_numer += digamma_ab * summand;
        sum_denom += summand;

        summand *= (1 + (a + b) / k) * (1 + k) / (1 + (a + 1) / k);
        digamma_ab += 1.0 / (a + b + k);
        ++k;
        summand *= z / k;

        if (k > 1e5)
          throw std::domain_error("stan::math::ddb_inc_beta did "
                                  "not converge within 100000 iterations");
      }

      return inc_beta(a, b, z)
             * (log(1 - z) - digamma_b + sum_numer / sum_denom);
    }

  }  // math
}   // stan

#endif
