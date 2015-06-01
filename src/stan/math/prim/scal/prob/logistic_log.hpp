#ifndef STAN_MATH_PRIM_SCAL_PROB_LOGISTIC_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_LOGISTIC_LOG_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1p.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/contains_nonconstant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <cmath>

namespace stan {
  namespace math {

    // Logistic(y|mu, sigma)    [sigma > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function("stan::math::logistic_log");
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::include_summand;
      using std::log;
      using std::exp;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_finite(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_loc, T_scale>::value)
        return 0.0;


      // set up template expressions wrapping scalars into vector views
      OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      VectorBuilder<true, T_partials_return, T_scale> inv_sigma(length(sigma));
      VectorBuilder<include_summand<propto, T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
        if (include_summand<propto, T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
      }

      VectorBuilder<!is_constant_struct<T_loc>::value,
                    T_partials_return, T_loc, T_scale>
        exp_mu_div_sigma(max_size(mu, sigma));
      VectorBuilder<!is_constant_struct<T_loc>::value,
                    T_partials_return, T_y, T_scale>
        exp_y_div_sigma(max_size(y, sigma));
      if (!is_constant_struct<T_loc>::value) {
        for (size_t n = 0; n < max_size(mu, sigma); n++)
          exp_mu_div_sigma[n] = exp(value_of(mu_vec[n])
                                    / value_of(sigma_vec[n]));
        for (size_t n = 0; n < max_size(y, sigma); n++)
          exp_y_div_sigma[n] = exp(value_of(y_vec[n])
                                   / value_of(sigma_vec[n]));
      }

      using stan::math::log1p;
      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);

        const T_partials_return y_minus_mu = y_dbl - mu_dbl;
        const T_partials_return y_minus_mu_div_sigma = y_minus_mu
          * inv_sigma[n];
        T_partials_return exp_m_y_minus_mu_div_sigma(0);
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          exp_m_y_minus_mu_div_sigma = exp(-y_minus_mu_div_sigma);
        T_partials_return inv_1p_exp_y_minus_mu_div_sigma(0);
        if (contains_nonconstant_struct<T_y, T_scale>::value)
          inv_1p_exp_y_minus_mu_div_sigma = 1 / (1 + exp(y_minus_mu_div_sigma));

        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          logp -= y_minus_mu_div_sigma;
        if (include_summand<propto, T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          logp -= 2.0 * log1p(exp_m_y_minus_mu_div_sigma);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n]
            += (2 * inv_1p_exp_y_minus_mu_div_sigma - 1) * inv_sigma[n];
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] +=
            (1 - 2 * exp_mu_div_sigma[n] / (exp_mu_div_sigma[n]
                                            + exp_y_div_sigma[n]))
            * inv_sigma[n];
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] +=
            ((1 - 2 * inv_1p_exp_y_minus_mu_div_sigma)
             *y_minus_mu*inv_sigma[n] - 1) * inv_sigma[n];
      }
      return operands_and_partials.to_var(logp, y, mu, sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y, T_loc, T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return logistic_log<false>(y, mu, sigma);
    }
  }
}
#endif
