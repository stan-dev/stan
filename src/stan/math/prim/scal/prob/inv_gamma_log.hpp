#ifndef STAN_MATH_PRIM_SCAL_PROB_INV_GAMMA_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_INV_GAMMA_LOG_HPP

#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_gamma.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <cmath>

namespace stan {

  namespace math {

    /**
     * The log of an inverse gamma density for y with the specified
     * shape and scale parameters.
     * Shape and scale parameters must be greater than 0.
     * y must be greater than 0.
     *
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_scale Type of scale.
     */
    template <bool propto,
              typename T_y, typename T_shape, typename T_scale>
    typename return_type<T_y, T_shape, T_scale>::type
    inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta) {
      static const char* function("stan::math::inv_gamma_log");
      typedef typename stan::partials_return_type<T_y, T_shape, T_scale>::type
        T_partials_return;

      using stan::is_constant_struct;
      using stan::math::check_not_nan;
      using stan::math::check_positive_finite;
      using boost::math::tools::promote_args;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(alpha)
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", beta);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Shape parameter", alpha,
                             "Scale parameter", beta);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_shape, T_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_scale> beta_vec(beta);

      for (size_t n = 0; n < length(y); n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        if (y_dbl <= 0)
          return LOG_ZERO;
      }

      size_t N = max_size(y, alpha, beta);
      OperandsAndPartials<T_y, T_shape, T_scale>
        operands_and_partials(y, alpha, beta);

      using stan::math::lgamma;
      using stan::math::digamma;
      using std::log;

      VectorBuilder<include_summand<propto, T_y, T_shape>::value,
                    T_partials_return, T_y> log_y(length(y));
      VectorBuilder<include_summand<propto, T_y, T_scale>::value,
                    T_partials_return, T_y> inv_y(length(y));
      for (size_t n = 0; n < length(y); n++) {
        if (include_summand<propto, T_y, T_shape>::value)
          if (value_of(y_vec[n]) > 0)
            log_y[n] = log(value_of(y_vec[n]));
        if (include_summand<propto, T_y, T_scale>::value)
          inv_y[n] = 1.0 / value_of(y_vec[n]);
      }

      VectorBuilder<include_summand<propto, T_shape>::value,
                    T_partials_return, T_shape> lgamma_alpha(length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape> digamma_alpha(length(alpha));
      for (size_t n = 0; n < length(alpha); n++) {
        if (include_summand<propto, T_shape>::value)
          lgamma_alpha[n] = lgamma(value_of(alpha_vec[n]));
        if (!is_constant_struct<T_shape>::value)
          digamma_alpha[n] = digamma(value_of(alpha_vec[n]));
      }

      VectorBuilder<include_summand<propto, T_shape, T_scale>::value,
                    T_partials_return, T_scale> log_beta(length(beta));
      if (include_summand<propto, T_shape, T_scale>::value) {
        for (size_t n = 0; n < length(beta); n++)
          log_beta[n] = log(value_of(beta_vec[n]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
        const T_partials_return beta_dbl = value_of(beta_vec[n]);

        if (include_summand<propto, T_shape>::value)
          logp -= lgamma_alpha[n];
        if (include_summand<propto, T_shape, T_scale>::value)
          logp += alpha_dbl * log_beta[n];
        if (include_summand<propto, T_y, T_shape>::value)
          logp -= (alpha_dbl+1.0) * log_y[n];
        if (include_summand<propto, T_y, T_scale>::value)
          logp -= beta_dbl * inv_y[n];

        // gradients
        if (!is_constant<typename is_vector<T_y>::type>::value)
          operands_and_partials.d_x1[n]
            += -(alpha_dbl+1) * inv_y[n] + beta_dbl * inv_y[n] * inv_y[n];
        if (!is_constant<typename is_vector<T_shape>::type>::value)
          operands_and_partials.d_x2[n]
            += -digamma_alpha[n] + log_beta[n] - log_y[n];
        if (!is_constant<typename is_vector<T_scale>::type>::value)
          operands_and_partials.d_x3[n] += alpha_dbl / beta_dbl - inv_y[n];
      }
      return operands_and_partials.to_var(logp, y, alpha, beta);
    }

    template <typename T_y, typename T_shape, typename T_scale>
    inline
    typename return_type<T_y, T_shape, T_scale>::type
    inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta) {
      return inv_gamma_log<false>(y, alpha, beta);
    }
  }
}

#endif
