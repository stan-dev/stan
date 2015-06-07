#ifndef STAN_MATH_PRIM_SCAL_PROB_EXP_MOD_NORMAL_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_EXP_MOD_NORMAL_LOG_HPP

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

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale,
              typename T_inv_scale>
    typename return_type<T_y, T_loc, T_scale, T_inv_scale>::type
    exp_mod_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
                       const T_inv_scale& lambda) {
      static const char* function("stan::math::exp_mod_normal_log");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale,
                                                  T_inv_scale>::type
        T_partials_return;

      using stan::is_constant_struct;
      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::include_summand;
      using std::log;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(sigma)
            && stan::length(lambda)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Inv_scale paramter", lambda);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_loc, T_scale, T_inv_scale>::value)
        return 0.0;

      using boost::math::erfc;
      using std::sqrt;
      using std::log;
      using std::exp;

      // set up template expressions wrapping scalars into vector views
      OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale>
        operands_and_partials(y, mu, sigma, lambda);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return lambda_dbl = value_of(lambda_vec[n]);

        const T_partials_return pi_dbl = boost::math::constants::pi<double>();

        // log probability
        if (include_summand<propto>::value)
          logp -= log(2.0);
        if (include_summand<propto, T_inv_scale>::value)
          logp += log(lambda_dbl);
        if (include_summand<propto, T_y, T_loc, T_scale, T_inv_scale>::value)
          logp += lambda_dbl
            * (mu_dbl + 0.5 * lambda_dbl * sigma_dbl * sigma_dbl - y_dbl)
            + log(erfc((mu_dbl + lambda_dbl * sigma_dbl
                        * sigma_dbl - y_dbl)
                       / (sqrt(2.0) * sigma_dbl)));

        // gradients
        const T_partials_return deriv_logerfc
          = -2.0 / sqrt(pi_dbl)
          * exp(-(mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl)
                / (std::sqrt(2.0) * sigma_dbl)
                * (mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl)
                / (sigma_dbl * std::sqrt(2.0)))
          / erfc((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl
                  - y_dbl) / (sigma_dbl * std::sqrt(2.0)));

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n]
            += -lambda_dbl
            + deriv_logerfc * -1.0 / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n]
            += lambda_dbl
            + deriv_logerfc / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n]
            += sigma_dbl * lambda_dbl * lambda_dbl
            + deriv_logerfc
            * (-mu_dbl / (sigma_dbl * sigma_dbl * std::sqrt(2.0))
               + lambda_dbl / std::sqrt(2.0)
               + y_dbl / (sigma_dbl * sigma_dbl * std::sqrt(2.0)));
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x4[n]
            += 1 / lambda_dbl + lambda_dbl * sigma_dbl * sigma_dbl
            + mu_dbl - y_dbl + deriv_logerfc * sigma_dbl / std::sqrt(2.0);
      }
      return operands_and_partials.to_var(logp, y, mu, sigma, lambda);
    }

    template <typename T_y, typename T_loc, typename T_scale,
              typename T_inv_scale>
    inline
    typename return_type<T_y, T_loc, T_scale, T_inv_scale>::type
    exp_mod_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
                       const T_inv_scale& lambda) {
      return exp_mod_normal_log<false>(y, mu, sigma, lambda);
    }
  }
}
#endif



