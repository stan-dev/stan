#ifndef STAN_MATH_PRIM_SCAL_PROB_WEIBULL_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_WEIBULL_LOG_HPP

#include <boost/random/weibull_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <cmath>

namespace stan {

  namespace math {

    // Weibull(y|alpha, sigma)     [y >= 0;  alpha > 0;  sigma > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_shape, typename T_scale>
    typename return_type<T_y, T_shape, T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      static const char* function("stan::math::weibull_log");
      typedef typename stan::partials_return_type<T_y, T_shape, T_scale>::type
        T_partials_return;

      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::math::multiply_log;
      using std::log;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(alpha)
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);
      check_finite(function, "Random variable", y);
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Shape parameter", alpha,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_shape, T_scale>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, alpha, sigma);

      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        if (y_dbl < 0)
          return LOG_ZERO;
      }

      VectorBuilder<include_summand<propto, T_shape>::value,
                    T_partials_return, T_shape> log_alpha(length(alpha));
      for (size_t i = 0; i < length(alpha); i++)
        if (include_summand<propto, T_shape>::value)
          log_alpha[i] = log(value_of(alpha_vec[i]));

      VectorBuilder<include_summand<propto, T_y, T_shape>::value,
                    T_partials_return, T_y> log_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto, T_y, T_shape>::value)
          log_y[i] = log(value_of(y_vec[i]));

      VectorBuilder<include_summand<propto, T_shape, T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++)
        if (include_summand<propto, T_shape, T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));

      VectorBuilder<include_summand<propto, T_y, T_shape, T_scale>::value,
                    T_partials_return, T_scale> inv_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++)
        if (include_summand<propto, T_y, T_shape, T_scale>::value)
          inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);

      VectorBuilder<include_summand<propto, T_y, T_shape, T_scale>::value,
                    T_partials_return, T_y, T_shape, T_scale>
        y_div_sigma_pow_alpha(N);
      for (size_t i = 0; i < N; i++)
        if (include_summand<propto, T_y, T_shape, T_scale>::value) {
          const T_partials_return y_dbl = value_of(y_vec[i]);
          const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
          y_div_sigma_pow_alpha[i] = pow(y_dbl * inv_sigma[i], alpha_dbl);
        }

      OperandsAndPartials<T_y, T_shape, T_scale>
        operands_and_partials(y, alpha, sigma);
      for (size_t n = 0; n < N; n++) {
        const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
        if (include_summand<propto, T_shape>::value)
          logp += log_alpha[n];
        if (include_summand<propto, T_y, T_shape>::value)
          logp += (alpha_dbl-1.0)*log_y[n];
        if (include_summand<propto, T_shape, T_scale>::value)
          logp -= alpha_dbl*log_sigma[n];
        if (include_summand<propto, T_y, T_shape, T_scale>::value)
          logp -= y_div_sigma_pow_alpha[n];

        if (!is_constant_struct<T_y>::value) {
          const T_partials_return inv_y = 1.0 / value_of(y_vec[n]);
          operands_and_partials.d_x1[n]
            += (alpha_dbl-1.0) * inv_y
            - alpha_dbl * y_div_sigma_pow_alpha[n] * inv_y;
        }
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n]
            += 1.0/alpha_dbl
            + (1.0 - y_div_sigma_pow_alpha[n]) * (log_y[n] - log_sigma[n]);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n]
            += alpha_dbl * inv_sigma[n] * (y_div_sigma_pow_alpha[n] - 1.0);
      }
      return operands_and_partials.to_var(logp, y, alpha, sigma);
    }

    template <typename T_y, typename T_shape, typename T_scale>
    inline
    typename return_type<T_y, T_shape, T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_log<false>(y, alpha, sigma);
    }
  }
}
#endif
