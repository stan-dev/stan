#ifndef STAN_MATH_PRIM_SCAL_FUN_REG_INC_BETA_DERIVS_HPP
#define STAN_MATH_PRIM_SCAL_FUN_REG_INC_BETA_DERIVS_HPP

#include <cmath>
#include <boost/math/special_functions/beta.hpp>

namespace stan {
  namespace math {

    // Gradients of the regularized incomplete beta function ibeta(a, b, z)
    template <typename T>
    double dda_grad_reg_inc_beta(T a, T b, T z,
                                 T digamma_a, T digamma_ab) {
      T threshold = 1e-8;
      
      digamma_a += 1.0 / a; // Need digamma(a + 1), not digamma(a);

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

        if (k > 1e4)
          throw std::domain_error("stan::math::dda_grad_reg_inc_beta did "
                                  "not converge within 10000 iterations");
      }

      return boost::math::ibeta(a, b, z)
      * (std::log(z) + sum_numer / sum_denom);
    }
    
    template <typename T>
    double ddb_grad_reg_inc_beta(T a, T b, T z,
                                 T digamma_b, T digamma_ab) {
      T threshold = 1e-8;

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

        if (k > 1e4)
          throw std::domain_error("stan::math::ddb_grad_reg_inc_beta did "
                                  "not converge within 10000 iterations");
      }

      return boost::math::ibeta(a, b, z)
      * (std::log(1 - z) - digamma_b + sum_numer / sum_denom);
    }

  } // math
}  // stan

#endif
