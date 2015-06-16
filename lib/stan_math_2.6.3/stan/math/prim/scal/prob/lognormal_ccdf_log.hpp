#ifndef STAN_MATH_PRIM_SCAL_PROB_LOGNORMAL_CCDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_LOGNORMAL_CCDF_LOG_HPP

#include <boost/random/lognormal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    lognormal_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function("stan::math::lognormal_ccdf_log");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      T_partials_return ccdf_log = 0.0;

      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive_finite;
      using boost::math::tools::promote_args;
      using stan::math::value_of;
      using std::log;
      using std::exp;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(sigma)))
        return ccdf_log;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      const double sqrt_pi = std::sqrt(stan::math::pi());

      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0.0)
          return operands_and_partials.to_var(0.0, y, mu, sigma);
      }

      const double log_half = std::log(0.5);

      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return scaled_diff = (log(y_dbl) - mu_dbl)
          / (sigma_dbl * SQRT_2);
        const T_partials_return rep_deriv = SQRT_2 / sqrt_pi
          * exp(-scaled_diff * scaled_diff) / sigma_dbl;

        // ccdf_log
        const T_partials_return erfc_calc = erfc(scaled_diff);
        ccdf_log += log_half + log(erfc_calc);

        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv / erfc_calc / y_dbl;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += rep_deriv / erfc_calc;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += rep_deriv * scaled_diff * SQRT_2
            / erfc_calc;
      }

      return operands_and_partials.to_var(ccdf_log, y, mu, sigma);
    }
  }
}
#endif
