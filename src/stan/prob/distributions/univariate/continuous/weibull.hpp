#ifndef __STAN__PROB__DISTRIBUTIONS__WEIBULL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__WEIBULL_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // Weibull(y|sigma,alpha)     [y >= 0;  sigma > 0;  alpha > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_shape, typename T_scale, 
              class Policy>
    typename return_type<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma, 
                const Policy&) {
      static const char* function = "stan::prob::weibull_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_positive;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      typename return_type<T_y,T_shape,T_scale>::type logp(0.0);
      if(!check_finite(function, y, "Random variable", &logp, Policy()))
        return logp;
      if(!check_finite(function, alpha, "Shape parameter", 
                       &logp, Policy()))
        return logp;
      if(!check_positive(function, alpha, "Shape parameter",
                         &logp, Policy()))
        return logp;
      if(!check_not_nan(function, sigma, "Scale parameter",
                        &logp, Policy()))
        return logp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,alpha,sigma,
				   "Random variable","Shape parameter","Scale parameter",
                                   &logp, Policy())))
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

      using stan::math::multiply_log;
      
      for (size_t n = 0; n < N; n++) {
	if (include_summand<propto,T_shape>::value)
	  logp += log(alpha_vec[n]);
	if (include_summand<propto,T_y,T_shape>::value)
	  logp += multiply_log(alpha_vec[n]-1.0, y_vec[n]);
	if (include_summand<propto,T_shape,T_scale>::value)
	  logp -= multiply_log(alpha_vec[n], sigma_vec[n]);
	if (include_summand<propto,T_y,T_shape,T_scale>::value)
	  logp -= pow(y_vec[n] / sigma_vec[n], alpha_vec[n]);
      }
      return logp;
    }


    template <bool propto,
              typename T_y, typename T_shape, typename T_scale>
    inline
    typename return_type<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_log<propto>(y,alpha,sigma,stan::math::default_policy());
    }


    template <typename T_y, typename T_shape, typename T_scale, 
              class Policy>
    inline
    typename return_type<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma, 
                const Policy&) {
      return weibull_log<false>(y,alpha,sigma,Policy());
    }


    template <typename T_y, typename T_shape, typename T_scale>
    inline
    typename return_type<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_log<false>(y,alpha,sigma,stan::math::default_policy());
    }




    template <typename T_y, typename T_shape, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_cdf(const T_y& y, const T_shape& alpha, const T_scale& sigma, 
              const Policy&) {

      static const char* function = "stan::prob::weibull_cdf(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_shape,T_scale>::type lp;
      if (!check_finite(function, alpha, "Shape parameter", 
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, "Shape parameter",
                          &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter",
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &lp, Policy()))
        return lp;
      
      if (y < 0.0)
        return 0.0;
      return 1.0 - exp(-pow(y / sigma, alpha));
    }

    template <typename T_y, typename T_shape, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_cdf(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_cdf(y,alpha,sigma,stan::math::default_policy());
    }

  }
}
#endif
