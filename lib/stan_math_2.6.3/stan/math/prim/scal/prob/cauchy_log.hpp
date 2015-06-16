#ifndef STAN_MATH_PRIM_SCAL_PROB_CAUCHY_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_CAUCHY_LOG_HPP

#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1p.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <cmath>

namespace stan {

  namespace math {

    /**
     * The log of the Cauchy density for the specified scalar(s) given
     * the specified location parameter(s) and scale parameter(s). y,
     * mu, or sigma can each either be scalar a vector.  Any vector
     * inputs must be the same length.
     *
     * <p> The result log probability is defined to be the sum of
     * the log probabilities for each observation/mu/sigma triple.
     *
     * @param y (Sequence of) scalar(s).
     * @param mu (Sequence of) location(s).
     * @param sigma (Sequence of) scale(s).
     * @return The log of the product of densities.
     * @tparam T_y Type of scalar outcome.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function("stan::math::cauchy_log");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      using stan::is_constant_struct;
      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_loc, T_scale>::value)
        return 0.0;

      using stan::math::log1p;
      using stan::math::square;
      using std::log;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      VectorBuilder<true, T_partials_return, T_scale> inv_sigma(length(sigma));
      VectorBuilder<true, T_partials_return,
                    T_scale> sigma_squared(length(sigma));
      VectorBuilder<include_summand<propto, T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        const T_partials_return sigma_dbl = value_of(sigma_vec[i]);
        inv_sigma[i] = 1.0 / sigma_dbl;
        sigma_squared[i] = sigma_dbl * sigma_dbl;
        if (include_summand<propto, T_scale>::value) {
          log_sigma[i] = log(sigma_dbl);
        }
      }

      OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);

        // reusable subexpression values
        const T_partials_return y_minus_mu
          = y_dbl - mu_dbl;
        const T_partials_return y_minus_mu_squared
          = y_minus_mu * y_minus_mu;
        const T_partials_return y_minus_mu_over_sigma
          = y_minus_mu * inv_sigma[n];
        const T_partials_return y_minus_mu_over_sigma_squared
          = y_minus_mu_over_sigma * y_minus_mu_over_sigma;

        // log probability
        if (include_summand<propto>::value)
          logp += NEG_LOG_PI;
        if (include_summand<propto, T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          logp -= log1p(y_minus_mu_over_sigma_squared);

        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= 2 * y_minus_mu
            / (sigma_squared[n] + y_minus_mu_squared);
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += 2 * y_minus_mu
            / (sigma_squared[n] + y_minus_mu_squared);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n]
            += (y_minus_mu_squared - sigma_squared[n])
            * inv_sigma[n] / (sigma_squared[n] + y_minus_mu_squared);
      }
      return operands_and_partials.to_var(logp, y, mu, sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y, T_loc, T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return cauchy_log<false>(y, mu, sigma);
    }


  }
}
#endif
