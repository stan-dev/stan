#ifndef STAN_MATH_PRIM_SCAL_PROB_NORMAL_SUFFICIENT_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_NORMAL_SUFFICIENT_LOG_HPP

#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/max_size.hpp>

namespace stan {

  namespace math {

    /**
     * The log of the normal density for the specified scalar(s) given
     * the specified mean(s) and deviation(s). y, mu, or sigma can
     * each be either a scalar or a std vector. Any vector inputs
     * must be the same length.
     *
     * <p>The result log probability is defined to be the sum of the
     * log probabilities for each observation/mean/deviation triple.
     * @param y (Sequence of) scalar(s).
     * @param mu (Sequence of) location parameter(s)
     * for the normal distribution.
     * @param sigma (Sequence of) scale parameters for the normal
     * distribution.
     * @return The log of the product of the densities.
     * @throw std::domain_error if the scale is not positive.
     * @tparam T_y Underlying type of scalar in sequence.
     * @tparam T_loc Type of location parameter.
     */
    template <bool propto,
              typename T_y, typename T_s, typename T_n, typename T_loc,
              typename T_scale>
    typename return_type<T_y, T_s, T_loc, T_scale>::type
    normal_sufficient_log(const T_y& y_bar, const T_s& s_squared,
                          const T_n& n_obs, const T_loc& mu,
                          const T_scale& sigma) {
      static const char* function = "stan::math::normal_log(%1%)";
      typedef typename
        stan::partials_return_type<T_y, T_s, T_n, T_loc, T_scale>::type
        T_partials_return;

      using std::log;
      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(y_bar)
            && stan::length(s_squared)
            && stan::length(n_obs)
            && stan::length(mu)
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function,
                    "Location parameter sufficient statistic", y_bar);
      check_not_nan(function,
                    "Scale parameter sufficient statistic", s_squared);
      check_not_nan(function,
                    "Number of observations", n_obs);
      check_finite(function,
                    "Number of observations", n_obs);
      check_positive(function,
                    "Number of observations", n_obs);
      check_finite(function,
                    "Location parameter", mu);
      check_positive(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Location parameter sufficient statistic",
                             y_bar,
                             "Scale parameter sufficient statistic",
                             s_squared,
                             "Number of observations", n_obs,
                             "Location parameter", mu,
                             "Scale parameter", sigma);
      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_s, T_loc, T_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      OperandsAndPartials<T_y, T_s, T_loc, T_scale>
        operands_and_partials(y_bar, s_squared, mu, sigma);

      VectorView<const T_y> y_bar_vec(y_bar);
      VectorView<const T_s> s_squared_vec(s_squared);
      VectorView<const T_n> n_obs_vec(n_obs);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y_bar, s_squared, n_obs, mu, sigma);

      for (size_t i = 0; i < N; i++) {
        const T_partials_return y_bar_dbl = value_of(y_bar_vec[i]);
        const T_partials_return s_squared_dbl =
          value_of(s_squared_vec[i]);
        const T_partials_return n_obs_dbl = n_obs_vec[i];
        const T_partials_return mu_dbl = value_of(mu_vec[i]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[i]);
        const T_partials_return sigma_squared = pow(sigma_dbl, 2);

        if (include_summand<propto>::value)
          logp += NEG_LOG_SQRT_TWO_PI * n_obs_dbl;

        if (include_summand<propto, T_scale>::value)
          logp -= n_obs_dbl*log(sigma_dbl);


        const T_partials_return cons_expr =
          (s_squared_dbl
           + n_obs_dbl * pow(y_bar_dbl - mu_dbl, 2));

        logp -= cons_expr / (2 * sigma_squared);


        // gradients
        if (!is_constant_struct<T_y>::value ||
            !is_constant_struct<T_loc>::value) {
          const T_partials_return common_derivative =
            n_obs_dbl * (mu_dbl - y_bar_dbl) / sigma_squared;
          if (!is_constant_struct<T_y>::value)
            operands_and_partials.d_x1[i] += common_derivative;
          if (!is_constant_struct<T_loc>::value)
            operands_and_partials.d_x3[i] -= common_derivative;
        }
        if (!is_constant_struct<T_s>::value)
          operands_and_partials.d_x2[i] -=
            1 / (2 * sigma_squared);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x4[i]
            += cons_expr / pow(sigma_dbl, 3) - n_obs_dbl / sigma_dbl;
      }
      return
        operands_and_partials.to_var(logp, y_bar, s_squared, mu, sigma);
    }

    template <typename T_y, typename T_s, typename T_n,
              typename T_loc, typename T_scale>
    inline
    typename return_type<T_y, T_s, T_loc, T_scale>::type
    normal_sufficient_log(const T_y& y_bar, const T_s& s_squared,
                          const T_n& n_obs, const T_loc& mu,
                          const T_scale& sigma) {
      return
        normal_sufficient_log<false>(y_bar, s_squared, n_obs, mu, sigma);
    }

  }
}
#endif
