#ifndef STAN_MATH_PRIM_SCAL_PROB_EXP_MOD_NORMAL_CDF_HPP
#define STAN_MATH_PRIM_SCAL_PROB_EXP_MOD_NORMAL_CDF_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <cmath>

namespace stan {

  namespace math {

    template <typename T_y, typename T_loc, typename T_scale,
              typename T_inv_scale>
    typename return_type<T_y, T_loc, T_scale, T_inv_scale>::type
    exp_mod_normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma,
                       const T_inv_scale& lambda) {
      static const char* function("stan::math::exp_mod_normal_cdf");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale,
                                                  T_inv_scale>::type
        T_partials_return;

      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      T_partials_return cdf(1.0);
      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(sigma)
            && stan::length(lambda)))
        return cdf;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive_finite(function, "Scale parameter", sigma);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_not_nan(function, "Inv_scale parameter", lambda);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Inv_scale paramter", lambda);

      OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale>
        operands_and_partials(y, mu, sigma, lambda);

      using stan::math::SQRT_2;
      using std::exp;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);
      const double sqrt_pi = std::sqrt(stan::math::pi());
      for (size_t n = 0; n < N; n++) {
        if (boost::math::isinf(y_vec[n])) {
          if (y_vec[n] < 0.0)
            return operands_and_partials.to_var(0.0, y, mu, sigma, lambda);
        }

        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return lambda_dbl = value_of(lambda_vec[n]);
        const T_partials_return u = lambda_dbl * (y_dbl - mu_dbl);
        const T_partials_return v = lambda_dbl * sigma_dbl;
        const T_partials_return v_sq = v * v;
        const T_partials_return scaled_diff = (y_dbl - mu_dbl) / (SQRT_2
                                                                  * sigma_dbl);
        const T_partials_return scaled_diff_sq = scaled_diff * scaled_diff;
        const T_partials_return erf_calc = 0.5 * (1 + erf(-v / SQRT_2
                                                          + scaled_diff));
        const T_partials_return deriv_1 = lambda_dbl * exp(0.5 * v_sq - u)
          * erf_calc;
        const T_partials_return deriv_2 = SQRT_2 / sqrt_pi * 0.5
          * exp(0.5 * v_sq - (scaled_diff - (v / SQRT_2))
                * (scaled_diff - (v / SQRT_2)) - u) / sigma_dbl;
        const T_partials_return deriv_3 = SQRT_2 / sqrt_pi * 0.5
          * exp(-scaled_diff_sq) / sigma_dbl;

        const T_partials_return cdf_ = 0.5 * (1 + erf(u / (v * SQRT_2)))
          - exp(-u + v_sq * 0.5) * (erf_calc);

        cdf *= cdf_;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += (deriv_1 - deriv_2 + deriv_3)
            / cdf_;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += (-deriv_1 + deriv_2 - deriv_3)
            / cdf_;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += (-deriv_1 * v - deriv_3
                                            * scaled_diff * SQRT_2 - deriv_2
                                            * sigma_dbl * SQRT_2
                                            * (-SQRT_2 * 0.5
                                               * (-lambda_dbl + scaled_diff
                                                  * SQRT_2 / sigma_dbl) - SQRT_2
                                               * lambda_dbl)) / cdf_;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x4[n] += exp(0.5 * v_sq - u)
            * (SQRT_2 / sqrt_pi * 0.5 * sigma_dbl
               * exp(-(v / SQRT_2 - scaled_diff) * (v / SQRT_2 - scaled_diff))
               - (v * sigma_dbl + mu_dbl - y_dbl) * erf_calc) / cdf_;
      }

      if (!is_constant_struct<T_y>::value) {
        for (size_t n = 0; n < stan::length(y); ++n)
          operands_and_partials.d_x1[n] *= cdf;
      }
      if (!is_constant_struct<T_loc>::value) {
        for (size_t n = 0; n < stan::length(mu); ++n)
          operands_and_partials.d_x2[n] *= cdf;
      }
      if (!is_constant_struct<T_scale>::value) {
        for (size_t n = 0; n < stan::length(sigma); ++n)
          operands_and_partials.d_x3[n] *= cdf;
      }
      if (!is_constant_struct<T_inv_scale>::value) {
        for (size_t n = 0; n < stan::length(lambda); ++n)
          operands_and_partials.d_x4[n] *= cdf;
      }

      return operands_and_partials.to_var(cdf, y, mu, sigma, lambda);
    }
  }
}
#endif



