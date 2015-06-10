#ifndef STAN_MATH_PRIM_SCAL_PROB_LOGNORMAL_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_LOGNORMAL_LOG_HPP

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
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/contains_nonconstant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <cmath>

namespace stan {
  namespace math {

    // LogNormal(y|mu, sigma)  [y >= 0;  sigma > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function("stan::math::lognormal_log");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      using stan::is_constant_struct;
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::include_summand;


      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      for (size_t n = 0; n < length(y); n++)
        if (value_of(y_vec[n]) <= 0)
          return LOG_ZERO;

      OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      using stan::math::square;
      using std::log;
      using stan::math::NEG_LOG_SQRT_TWO_PI;
      using std::log;


      VectorBuilder<include_summand<propto, T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      if (include_summand<propto, T_scale>::value) {
        for (size_t n = 0; n < length(sigma); n++)
          log_sigma[n] = log(value_of(sigma_vec[n]));
      }

      VectorBuilder<include_summand<propto, T_y, T_loc, T_scale>::value,
                    T_partials_return, T_scale> inv_sigma(length(sigma));
      VectorBuilder<include_summand<propto, T_y, T_loc, T_scale>::value,
                    T_partials_return, T_scale> inv_sigma_sq(length(sigma));
      if (include_summand<propto, T_y, T_loc, T_scale>::value) {
        for (size_t n = 0; n < length(sigma); n++)
          inv_sigma[n] = 1 / value_of(sigma_vec[n]);
      }
      if (include_summand<propto, T_y, T_loc, T_scale>::value) {
        for (size_t n = 0; n < length(sigma); n++)
          inv_sigma_sq[n] = inv_sigma[n] * inv_sigma[n];
      }

      VectorBuilder<include_summand<propto, T_y, T_loc, T_scale>::value,
                    T_partials_return, T_y> log_y(length(y));
      if (include_summand<propto, T_y, T_loc, T_scale>::value) {
        for (size_t n = 0; n < length(y); n++)
          log_y[n] = log(value_of(y_vec[n]));
      }

      VectorBuilder<!is_constant_struct<T_y>::value,
                    T_partials_return, T_y> inv_y(length(y));
      if (!is_constant_struct<T_y>::value) {
        for (size_t n = 0; n < length(y); n++)
          inv_y[n] = 1 / value_of(y_vec[n]);
      }

      if (include_summand<propto>::value)
        logp += N * NEG_LOG_SQRT_TWO_PI;

      for (size_t n = 0; n < N; n++) {
        const T_partials_return mu_dbl = value_of(mu_vec[n]);

        T_partials_return logy_m_mu(0);
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          logy_m_mu = log_y[n] - mu_dbl;

        T_partials_return logy_m_mu_sq = logy_m_mu * logy_m_mu;
        T_partials_return logy_m_mu_div_sigma(0);
        if (contains_nonconstant_struct<T_y, T_loc, T_scale>::value)
          logy_m_mu_div_sigma = logy_m_mu * inv_sigma_sq[n];


        // log probability
        if (include_summand<propto, T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto, T_y>::value)
          logp -= log_y[n];
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          logp -= 0.5 * logy_m_mu_sq * inv_sigma_sq[n];

        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= (1 + logy_m_mu_div_sigma) * inv_y[n];
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += logy_m_mu_div_sigma;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n]
            += (logy_m_mu_div_sigma * logy_m_mu - 1) * inv_sigma[n];
      }
      return operands_and_partials.to_var(logp, y, mu, sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y, T_loc, T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return lognormal_log<false>(y, mu, sigma);
    }
  }
}
#endif
