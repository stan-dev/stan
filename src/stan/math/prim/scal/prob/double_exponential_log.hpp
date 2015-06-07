#ifndef STAN_MATH_PRIM_SCAL_PROB_DOUBLE_EXPONENTIAL_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_DOUBLE_EXPONENTIAL_LOG_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/meta/contains_nonconstant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/sign.hpp>
#include <cmath>

namespace stan {

  namespace math {

    // DoubleExponential(y|mu, sigma)  [sigma > 0]
    // FIXME: add documentation
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    double_exponential_log(const T_y& y,
                           const T_loc& mu, const T_scale& sigma) {
      static const char* function("stan::math::double_exponential_log");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      using stan::is_constant_struct;
      using stan::math::check_finite;
      using stan::math::check_positive_finite;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::include_summand;
      using std::log;
      using std::fabs;
      using stan::math::sign;
      using std::log;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);
      check_finite(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Shape parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_loc, T_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      VectorBuilder<include_summand<propto, T_y, T_loc, T_scale>::value,
                    T_partials_return, T_scale> inv_sigma(length(sigma));
      VectorBuilder<!is_constant_struct<T_scale>::value,
                    T_partials_return, T_scale>
        inv_sigma_squared(length(sigma));
      VectorBuilder<include_summand<propto, T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        const T_partials_return sigma_dbl = value_of(sigma_vec[i]);
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          inv_sigma[i] = 1.0 / sigma_dbl;
        if (include_summand<propto, T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
        if (!is_constant_struct<T_scale>::value)
          inv_sigma_squared[i] = inv_sigma[i] * inv_sigma[i];
      }


      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);

        // reusable subexpressions values
        const T_partials_return y_m_mu = y_dbl - mu_dbl;
        const T_partials_return fabs_y_m_mu = fabs(y_m_mu);

        // log probability
        if (include_summand<propto>::value)
          logp += NEG_LOG_TWO;
        if (include_summand<propto, T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          logp -= fabs_y_m_mu * inv_sigma[n];

        // gradients
        T_partials_return sign_y_m_mu_times_inv_sigma(0);
        if (contains_nonconstant_struct<T_y, T_loc>::value)
          sign_y_m_mu_times_inv_sigma = sign(y_m_mu) * inv_sigma[n];
        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n] -= sign_y_m_mu_times_inv_sigma;
        }
        if (!is_constant_struct<T_loc>::value) {
          operands_and_partials.d_x2[n] += sign_y_m_mu_times_inv_sigma;
        }
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += -inv_sigma[n] + fabs_y_m_mu
            * inv_sigma_squared[n];
      }
      return operands_and_partials.to_var(logp, y, mu, sigma);
    }


    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    double_exponential_log(const T_y& y, const T_loc& mu,
                           const T_scale& sigma) {
      return double_exponential_log<false>(y, mu, sigma);
    }
  }
}
#endif
