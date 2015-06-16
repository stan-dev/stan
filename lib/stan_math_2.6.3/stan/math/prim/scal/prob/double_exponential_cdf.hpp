#ifndef STAN_MATH_PRIM_SCAL_PROB_DOUBLE_EXPONENTIAL_CDF_HPP
#define STAN_MATH_PRIM_SCAL_PROB_DOUBLE_EXPONENTIAL_CDF_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/sign.hpp>
#include <cmath>

namespace stan {

  namespace math {

    /**
     * Calculates the double exponential cumulative density function.
     *
     * \f$ f(y|\mu, \sigma) = \begin{cases} \
     \frac{1}{2} \exp\left(\frac{y-\mu}{\sigma}\right), \mbox{if } y < \mu \\
     1 - \frac{1}{2} \exp\left(-\frac{y-\mu}{\sigma}\right), \mbox{if } y \ge \mu \
     \end{cases}\f$
     *
     * @param y A scalar variate.
     * @param mu The location parameter.
     * @param sigma The scale parameter.
     *
     * @return The cumulative density function.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    double_exponential_cdf(const T_y& y,
                           const T_loc& mu, const T_scale& sigma) {
      static const char* function("stan::math::double_exponential_cdf");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      // Size checks
      if ( !( stan::length(y) && stan::length(mu)
              && stan::length(sigma) ) )
        return 1.0;

      using stan::math::value_of;
      using stan::math::check_finite;
      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      using std::exp;

      T_partials_return cdf(1.0);

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      // cdf
      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return scaled_diff = (y_dbl - mu_dbl) / (sigma_dbl);
        const T_partials_return exp_scaled_diff = exp(scaled_diff);

        if (y_dbl < mu_dbl)
          cdf *= exp_scaled_diff * 0.5;
        else
          cdf *= 1.0 - 0.5 / exp_scaled_diff;
      }

      // gradients
      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return scaled_diff = (y_dbl - mu_dbl) / sigma_dbl;
        const T_partials_return exp_scaled_diff = exp(scaled_diff);
        const T_partials_return inv_sigma = 1.0 / sigma_dbl;

        if (y_dbl < mu_dbl) {
          if (!is_constant_struct<T_y>::value)
            operands_and_partials.d_x1[n] += inv_sigma * cdf;
          if (!is_constant_struct<T_loc>::value)
            operands_and_partials.d_x2[n] -= inv_sigma * cdf;
          if (!is_constant_struct<T_scale>::value)
            operands_and_partials.d_x3[n] -= scaled_diff * inv_sigma  * cdf;
        } else {
          const T_partials_return rep_deriv = cdf * inv_sigma
            / (2.0 * exp_scaled_diff - 1.0);
          if (!is_constant_struct<T_y>::value)
            operands_and_partials.d_x1[n] += rep_deriv;
          if (!is_constant_struct<T_loc>::value)
            operands_and_partials.d_x2[n] -= rep_deriv;
          if (!is_constant_struct<T_scale>::value)
            operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff;
        }
      }
      return operands_and_partials.to_var(cdf, y, mu, sigma);
    }
  }
}
#endif
