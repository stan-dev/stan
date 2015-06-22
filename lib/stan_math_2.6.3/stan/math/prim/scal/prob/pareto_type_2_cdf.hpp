#ifndef STAN_MATH_PRIM_SCAL_PROB_PARETO_TYPE_2_CDF_HPP
#define STAN_MATH_PRIM_SCAL_PROB_PARETO_TYPE_2_CDF_HPP

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
#include <stan/math/prim/scal/meta/contains_nonconstant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y, T_loc, T_scale, T_shape>::type
    pareto_type_2_cdf(const T_y& y, const T_loc& mu,
                      const T_scale& lambda, const T_shape& alpha) {
      typedef
        typename stan::partials_return_type<T_y, T_loc, T_scale, T_shape>::type
        T_partials_return;

      // Check sizes
      // Size checks
      if ( !( stan::length(y)
              && stan::length(mu)
              && stan::length(lambda)
              && stan::length(alpha) ) )
        return 1.0;

      // Check errors
      static const char* function("stan::math::pareto_type_2_cdf");

      using stan::math::check_greater_or_equal;
      using stan::math::check_finite;
      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_greater_or_equal;
      using stan::math::check_consistent_sizes;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using std::log;

      T_partials_return P(1.0);

      check_greater_or_equal(function, "Random variable", y, mu);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", lambda);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Scale parameter", lambda,
                             "Shape parameter", alpha);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> lambda_vec(lambda);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, lambda, alpha);

      OperandsAndPartials<T_y, T_loc, T_scale, T_shape>
        operands_and_partials(y, mu, lambda, alpha);

      VectorBuilder<true, T_partials_return,
                    T_y, T_loc, T_scale, T_shape>
        p1_pow_alpha(N);

      VectorBuilder<contains_nonconstant_struct<T_y, T_loc, T_scale>::value,
                    T_partials_return, T_y, T_loc, T_scale, T_shape>
        grad_1_2(N);

      VectorBuilder<contains_nonconstant_struct<T_shape, T_y>::value,
                    T_partials_return, T_y, T_loc, T_scale, T_shape>
        grad_3(N);

      for (size_t i = 0; i < N; i++) {
        const T_partials_return lambda_dbl = value_of(lambda_vec[i]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
        const T_partials_return temp = 1 + (value_of(y_vec[i])
                                            - value_of(mu_vec[i]))
          / lambda_dbl;
        p1_pow_alpha[i] = pow(temp, -alpha_dbl);

        if (contains_nonconstant_struct<T_y, T_loc, T_scale>::value)
          grad_1_2[i] = p1_pow_alpha[i] / temp * alpha_dbl / lambda_dbl;

        if (!is_constant_struct<T_shape>::value)
          grad_3[i] = log(temp) * p1_pow_alpha[i];
      }

      // Compute vectorized CDF and its gradients

      for (size_t n = 0; n < N; n++) {
        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return lambda_dbl = value_of(lambda_vec[n]);

        const T_partials_return Pn = 1.0 - p1_pow_alpha[n];

        // Compute
        P *= Pn;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += grad_1_2[n] / Pn;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= grad_1_2[n] / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += (mu_dbl - y_dbl)
            * grad_1_2[n] / lambda_dbl / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] += grad_3[n] / Pn;
      }

      if (!is_constant_struct<T_y>::value) {
        for (size_t n = 0; n < stan::length(y); ++n)
          operands_and_partials.d_x1[n] *= P;
      }
      if (!is_constant_struct<T_loc>::value) {
        for (size_t n = 0; n < stan::length(mu); ++n)
          operands_and_partials.d_x2[n] *= P;
      }
      if (!is_constant_struct<T_scale>::value) {
        for (size_t n = 0; n < stan::length(lambda); ++n)
          operands_and_partials.d_x3[n] *= P;
      }
      if (!is_constant_struct<T_shape>::value) {
        for (size_t n = 0; n < stan::length(alpha); ++n)
          operands_and_partials.d_x4[n] *= P;
      }

      return operands_and_partials.to_var(P, y, mu, lambda, alpha);
    }
  }
}
#endif
