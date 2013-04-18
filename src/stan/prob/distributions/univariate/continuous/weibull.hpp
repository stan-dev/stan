#ifndef __STAN__PROB__DISTRIBUTIONS__WEIBULL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__WEIBULL_HPP__

#include <boost/random/weibull_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // Weibull(y|sigma,alpha)     [y >= 0;  sigma > 0;  alpha > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_shape, typename T_scale>
    typename return_type<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      static const char* function = "stan::prob::weibull_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_positive;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::math::multiply_log;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);
      if(!check_finite(function, y, "Random variable", &logp))
        return logp;
      if(!check_finite(function, alpha, "Shape parameter", 
                       &logp))
        return logp;
      if(!check_positive(function, alpha, "Shape parameter",
                         &logp))
        return logp;
      if(!check_not_nan(function, sigma, "Scale parameter",
                        &logp))
        return logp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,alpha,sigma,
                                   "Random variable","Shape parameter","Scale parameter",
                                   &logp)))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_shape,T_scale>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, alpha, sigma);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        if (y_dbl < 0)
          return LOG_ZERO;
      }
      
      DoubleVectorView<include_summand<propto,T_shape>::value,
        is_vector<T_shape>::value> log_alpha(length(alpha));
      for (size_t i = 0; i < length(alpha); i++)
        if (include_summand<propto,T_shape>::value)
          log_alpha[i] = log(value_of(alpha_vec[i]));
      
      DoubleVectorView<include_summand<propto,T_y,T_shape>::value,
        is_vector<T_y>::value> log_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y,T_shape>::value)
          log_y[i] = log(value_of(y_vec[i]));

      DoubleVectorView<include_summand<propto,T_shape,T_scale>::value,
        is_vector<T_scale>::value> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++)
        if (include_summand<propto,T_shape,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));

      DoubleVectorView<include_summand<propto,T_y,T_shape,T_scale>::value,
        is_vector<T_scale>::value> inv_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++)
        if (include_summand<propto,T_y,T_shape,T_scale>::value)
          inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
      
      DoubleVectorView<include_summand<propto,T_y,T_shape,T_scale>::value,
        is_vector<T_y>::value | is_vector<T_shape>::value | is_vector<T_scale>::value>
        y_div_sigma_pow_alpha(N);
      for (size_t i = 0; i < N; i++)
        if (include_summand<propto,T_y,T_shape,T_scale>::value) {
          const double y_dbl = value_of(y_vec[i]);
          const double alpha_dbl = value_of(alpha_vec[i]);
          y_div_sigma_pow_alpha[i] = pow(y_dbl * inv_sigma[i], alpha_dbl);
        }

      agrad::OperandsAndPartials<T_y,T_shape,T_scale> operands_and_partials(y,alpha,sigma);
      for (size_t n = 0; n < N; n++) {
        const double alpha_dbl = value_of(alpha_vec[n]);
        if (include_summand<propto,T_shape>::value)
          logp += log_alpha[n];
        if (include_summand<propto,T_y,T_shape>::value)
          logp += (alpha_dbl-1.0)*log_y[n];
        if (include_summand<propto,T_shape,T_scale>::value)
          logp -= alpha_dbl*log_sigma[n];
        if (include_summand<propto,T_y,T_shape,T_scale>::value)
          logp -= y_div_sigma_pow_alpha[n];

        if (!is_constant_struct<T_y>::value) {
          const double inv_y = 1.0 / value_of(y_vec[n]);
          operands_and_partials.d_x1[n] 
            += (alpha_dbl-1.0) * inv_y 
            - alpha_dbl * y_div_sigma_pow_alpha[n] * inv_y;
        }
        if (!is_constant_struct<T_shape>::value) 
          operands_and_partials.d_x2[n] 
            += 1.0/alpha_dbl 
            + (1.0 - y_div_sigma_pow_alpha[n]) * (log_y[n] - log_sigma[n]);
        if (!is_constant_struct<T_scale>::value) 
          operands_and_partials.d_x3[n] 
            += -alpha_dbl * inv_sigma[n]
            + alpha_dbl * inv_sigma[n] * y_div_sigma_pow_alpha[n];
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_shape, typename T_scale>
    inline
    typename return_type<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_log<false>(y,alpha,sigma);
    }

    template <typename T_y, typename T_shape, typename T_scale>
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_cdf(const T_y& y, const T_shape& alpha, const T_scale& sigma) {

      static const char* function = "stan::prob::weibull_cdf(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_shape,T_scale>::type lp;
      if (!check_finite(function, alpha, "Shape parameter", 
                        &lp))
        return lp;
      if (!check_positive(function, alpha, "Shape parameter",
                          &lp))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter",
                        &lp))
        return lp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &lp))
        return lp;
      
      if (y < 0.0)
        return 0.0;
      return 1.0 - exp(-pow(y / sigma, alpha));
    }
   
    template <class RNG>
    inline double
    weibull_rng(const double alpha,
                const double sigma,
                RNG& rng) {
      using boost::variate_generator;
      using boost::random::weibull_distribution;
      variate_generator<RNG&, weibull_distribution<> >
        weibull_rng(rng, weibull_distribution<>(alpha, sigma));
      return weibull_rng();
    }
  }
}
#endif
