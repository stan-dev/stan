#ifndef STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_CDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_CDF_LOG_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <cmath>
#include <limits>

namespace stan {

  namespace math {

    template <typename T_n, typename T_shape,
              typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_cdf_log(const T_n& n, const T_shape& alpha,
                         const T_inv_scale& beta) {
      static const char* function("stan::math::neg_binomial_cdf_log");
      typedef typename stan::partials_return_type<T_n, T_shape,
                                                  T_inv_scale>::type
        T_partials_return;

      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_consistent_sizes;
      using stan::math::include_summand;

      // Ensure non-zero arugment lengths
      if (!(stan::length(n) && stan::length(alpha) && stan::length(beta)))
        return 0.0;

      T_partials_return P(0.0);

      // Validate arguments
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Inverse scale parameter", beta);
      check_consistent_sizes(function,
                             "Failures variable", n,
                             "Shape parameter", alpha,
                             "Inverse scale parameter", beta);

      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);

      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      using stan::math::inc_beta;
      using stan::math::digamma;
      using stan::math::lbeta;
      using std::exp;
      using std::pow;
      using std::log;
      using std::exp;


      OperandsAndPartials<T_shape, T_inv_scale>
        operands_and_partials(alpha, beta);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0)
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              alpha, beta);
      }

      // Cache a few expensive function calls if alpha is a parameter
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape>
        digammaN_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape>
        digammaAlpha_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape>
        digammaSum_vec(stan::length(alpha));

      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const T_partials_return n_dbl = value_of(n_vec[i]);
          const T_partials_return alpha_dbl = value_of(alpha_vec[i]);

          digammaN_vec[i] = digamma(n_dbl + 1);
          digammaAlpha_vec[i] = digamma(alpha_dbl);
          digammaSum_vec[i] = digamma(n_dbl + alpha_dbl + 1);
        }
      }

      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) == std::numeric_limits<int>::max())
          return operands_and_partials.to_var(0.0, alpha, beta);

        const T_partials_return n_dbl = value_of(n_vec[i]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
        const T_partials_return beta_dbl = value_of(beta_vec[i]);
        const T_partials_return p_dbl = beta_dbl / (1.0 + beta_dbl);
        const T_partials_return d_dbl = 1.0 / ( (1.0 + beta_dbl)
                                                * (1.0 + beta_dbl) );
        const T_partials_return Pi = inc_beta(alpha_dbl, n_dbl + 1.0, p_dbl);
        const T_partials_return beta_func = exp(lbeta(n_dbl + 1, alpha_dbl));


        P += log(Pi);

        if (!is_constant_struct<T_shape>::value) {
          T_partials_return g1 = 0;
          T_partials_return g2 = 0;

          stan::math::grad_reg_inc_beta(g1, g2, alpha_dbl,
                                        n_dbl + 1, p_dbl,
                                        digammaAlpha_vec[i],
                                        digammaN_vec[i],
                                        digammaSum_vec[i],
                                        beta_func);
          operands_and_partials.d_x1[i] += g1 / Pi;
        }
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]  += d_dbl * pow(1-p_dbl, n_dbl)
            * pow(p_dbl, alpha_dbl-1) / beta_func / Pi;
      }

      return operands_and_partials.to_var(P, alpha, beta);
    }
  }
}
#endif
