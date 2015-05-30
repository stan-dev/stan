#ifndef STAN_MATH_PRIM_SCAL_PROB_PARETO_TYPE_2_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_PARETO_TYPE_2_LOG_HPP

#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <cmath>


namespace stan {
  namespace math {

    // pareto_type_2(y|lambda, alpha)  [y >= 0;  lambda > 0;  alpha > 0]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y, T_loc, T_scale, T_shape>::type
    pareto_type_2_log(const T_y& y, const T_loc& mu, const T_scale& lambda,
                      const T_shape& alpha) {
      static const char* function("stan::math::pareto_type_2_log");
      typedef
        typename stan::partials_return_type<T_y, T_loc, T_scale, T_shape>::type
        T_partials_return;

      using std::log;
      using stan::math::value_of;
      using stan::math::check_finite;
      using stan::math::check_greater_or_equal;
      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using std::log;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(lambda)
            && stan::length(alpha)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_greater_or_equal(function, "Random variable", y, mu);
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", lambda);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Scale parameter", lambda,
                             "Shape parameter", alpha);


      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_loc, T_scale, T_shape>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> lambda_vec(lambda);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, lambda, alpha);

      // set up template expressions wrapping scalars into vector views
      OperandsAndPartials<T_y, T_loc, T_scale, T_shape>
        operands_and_partials(y, mu, lambda, alpha);

      VectorBuilder<include_summand<propto, T_y, T_loc, T_scale, T_shape>
                    ::value,
                    T_partials_return, T_y, T_loc, T_scale>
        log1p_scaled_diff(N);
      if (include_summand<propto, T_y, T_loc, T_scale, T_shape>::value) {
        for (size_t n = 0; n < N; n++)
          log1p_scaled_diff[n] = log1p((value_of(y_vec[n])
                                        - value_of(mu_vec[n]))
                                       / value_of(lambda_vec[n]));
      }

      VectorBuilder<include_summand<propto, T_scale>::value,
                    T_partials_return, T_scale> log_lambda(length(lambda));
      if (include_summand<propto, T_scale>::value) {
        for (size_t n = 0; n < length(lambda); n++)
          log_lambda[n] = log(value_of(lambda_vec[n]));
      }

      VectorBuilder<include_summand<propto, T_shape>::value,
                    T_partials_return, T_shape> log_alpha(length(alpha));
      if (include_summand<propto, T_shape>::value) {
        for (size_t n = 0; n < length(alpha); n++)
          log_alpha[n] = log(value_of(alpha_vec[n]));
      }

      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape> inv_alpha(length(alpha));
      if (!is_constant_struct<T_shape>::value) {
        for (size_t n = 0; n < length(alpha); n++)
          inv_alpha[n] = 1 / value_of(alpha_vec[n]);
      }

      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return lambda_dbl = value_of(lambda_vec[n]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
        const T_partials_return sum_dbl = lambda_dbl + y_dbl - mu_dbl;
        const T_partials_return inv_sum = 1.0 / sum_dbl;
        const T_partials_return alpha_div_sum = alpha_dbl / sum_dbl;
        const T_partials_return deriv_1_2 = inv_sum + alpha_div_sum;

        // // log probability
        if (include_summand<propto, T_shape>::value)
          logp += log_alpha[n];
        if (include_summand<propto, T_scale>::value)
          logp -= log_lambda[n];
        if (include_summand<propto, T_y, T_scale, T_shape>::value)
          logp -= (alpha_dbl + 1.0) * log1p_scaled_diff[n];

        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= deriv_1_2;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += deriv_1_2;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= alpha_div_sum * (mu_dbl - y_dbl)
            / lambda_dbl + inv_sum;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] += inv_alpha[n] - log1p_scaled_diff[n];
      }
      return operands_and_partials.to_var(logp, y, mu, lambda, alpha);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline
    typename return_type<T_y, T_loc, T_scale, T_shape>::type
    pareto_type_2_log(const T_y& y, const T_loc& mu,
                      const T_scale& lambda, const T_shape& alpha) {
      return pareto_type_2_log<false>(y, mu, lambda, alpha);
    }
  }
}
#endif
