#ifndef STAN_MATH_PRIM_SCAL_PROB_BETA_CDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_BETA_CDF_LOG_HPP

#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/meta/contains_nonconstant_struct.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <cmath>

namespace stan {

  namespace math {

    template <typename T_y, typename T_scale_succ, typename T_scale_fail>
    typename return_type<T_y, T_scale_succ, T_scale_fail>::type
    beta_cdf_log(const T_y& y, const T_scale_succ& alpha,
                 const T_scale_fail& beta) {
      typedef typename stan::partials_return_type<T_y, T_scale_succ,
                                                  T_scale_fail>::type
        T_partials_return;

      // Size checks
      if ( !( stan::length(y) && stan::length(alpha)
              && stan::length(beta) ) )
        return 0.0;

      // Error checks
      static const char* function("stan::math::beta_cdf");

      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::check_less_or_equal;
      using boost::math::tools::promote_args;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      T_partials_return cdf_log(0.0);

      check_positive_finite(function, "First shape parameter", alpha);
      check_positive_finite(function, "Second shape parameter", beta);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_less_or_equal(function, "Random variable", y, 1);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "First shape parameter", alpha,
                             "Second shape parameter", beta);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale_succ> alpha_vec(alpha);
      VectorView<const T_scale_fail> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);

      OperandsAndPartials<T_y, T_scale_succ, T_scale_fail>
        operands_and_partials(y, alpha, beta);

      // Compute CDF and its gradients
      using stan::math::inc_beta;
      using stan::math::digamma;
      using stan::math::lbeta;
      using std::pow;
      using std::exp;
      using std::log;
      using std::exp;

      // Cache a few expensive function calls if alpha or beta is a parameter
      VectorBuilder<contains_nonconstant_struct<T_scale_succ,
                                                T_scale_fail>::value,
                    T_partials_return, T_scale_succ, T_scale_fail>
        digamma_alpha_vec(max_size(alpha, beta));

      VectorBuilder<contains_nonconstant_struct<T_scale_succ,
                                                T_scale_fail>::value,
                    T_partials_return, T_scale_succ, T_scale_fail>
        digamma_beta_vec(max_size(alpha, beta));

      VectorBuilder<contains_nonconstant_struct<T_scale_succ,
                                                T_scale_fail>::value,
                    T_partials_return, T_scale_succ, T_scale_fail>
        digamma_sum_vec(max_size(alpha, beta));

      if (contains_nonconstant_struct<T_scale_succ, T_scale_fail>::value) {
        for (size_t i = 0; i < N; i++) {
          const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
          const T_partials_return beta_dbl = value_of(beta_vec[i]);

          digamma_alpha_vec[i] = digamma(alpha_dbl);
          digamma_beta_vec[i] = digamma(beta_dbl);
          digamma_sum_vec[i] = digamma(alpha_dbl + beta_dbl);
        }
      }

      // Compute vectorized CDFLog and gradient
      for (size_t n = 0; n < N; n++) {
        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
        const T_partials_return beta_dbl = value_of(beta_vec[n]);
        const T_partials_return betafunc_dbl = exp(lbeta(alpha_dbl, beta_dbl));
        // Compute
        const T_partials_return Pn = inc_beta(alpha_dbl, beta_dbl, y_dbl);

        cdf_log += log(Pn);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += pow(1-y_dbl, beta_dbl-1)
            * pow(y_dbl, alpha_dbl-1) / betafunc_dbl / Pn;

        T_partials_return g1 = 0;
        T_partials_return g2 = 0;

        if (contains_nonconstant_struct<T_scale_succ, T_scale_fail>::value) {
          stan::math::grad_reg_inc_beta(g1, g2, alpha_dbl, beta_dbl, y_dbl,
                                        digamma_alpha_vec[n],
                                        digamma_beta_vec[n], digamma_sum_vec[n],
                                        betafunc_dbl);
        }
        if (!is_constant_struct<T_scale_succ>::value)
          operands_and_partials.d_x2[n] += g1 / Pn;
        if (!is_constant_struct<T_scale_fail>::value)
          operands_and_partials.d_x3[n]  += g2 / Pn;
      }

      return operands_and_partials.to_var(cdf_log, y, alpha, beta);
    }

  }
}
#endif
