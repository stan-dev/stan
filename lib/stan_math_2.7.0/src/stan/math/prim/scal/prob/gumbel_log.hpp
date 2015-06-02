#ifndef STAN_MATH_PRIM_SCAL_PROB_GUMBEL_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_GUMBEL_LOG_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <cmath>

namespace stan {

  namespace math {

    template <bool propto, typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    gumbel_log(const T_y& y, const T_loc& mu, const T_scale& beta) {
      static const char* function("stan::math::gumbel_log");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      using std::log;
      using std::exp;
      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::include_summand;
      using std::log;
      using std::exp;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive(function, "Scale parameter", beta);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", beta);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_loc, T_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> beta_vec(beta);
      size_t N = max_size(y, mu, beta);

      VectorBuilder<true, T_partials_return, T_scale> inv_beta(length(beta));
      VectorBuilder<include_summand<propto, T_scale>::value,
                    T_partials_return, T_scale> log_beta(length(beta));
      for (size_t i = 0; i < length(beta); i++) {
        inv_beta[i] = 1.0 / value_of(beta_vec[i]);
        if (include_summand<propto, T_scale>::value)
          log_beta[i] = log(value_of(beta_vec[i]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);

        // reusable subexpression values
        const T_partials_return y_minus_mu_over_beta
          = (y_dbl - mu_dbl) * inv_beta[n];

        // log probability
        if (include_summand<propto, T_scale>::value)
          logp -= log_beta[n];
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          logp += -y_minus_mu_over_beta - exp(-y_minus_mu_over_beta);

        // gradients
        T_partials_return scaled_diff = inv_beta[n]
          * exp(-y_minus_mu_over_beta);
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= inv_beta[n] - scaled_diff;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += inv_beta[n] - scaled_diff;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n]
            += -inv_beta[n] + y_minus_mu_over_beta * inv_beta[n]
            - scaled_diff * y_minus_mu_over_beta;
      }
      return operands_and_partials.to_var(logp, y, mu, beta);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y, T_loc, T_scale>::type
    gumbel_log(const T_y& y, const T_loc& mu, const T_scale& beta) {
      return gumbel_log<false>(y, mu, beta);
    }
  }
}
#endif

