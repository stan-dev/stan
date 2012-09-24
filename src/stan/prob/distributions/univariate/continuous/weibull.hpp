#ifndef __STAN__PROB__DISTRIBUTIONS__WEIBULL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__WEIBULL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // Weibull(y|sigma,alpha)     [y >= 0;  sigma > 0;  alpha > 0]
    template <bool propto,
              typename T_y, typename T_shape, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma, 
                const Policy&) {
      static const char* function = "stan::prob::weibull_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_shape,T_scale>::type lp = 0.0;
      if(!check_finite(function, y, "Random variable", &lp, Policy()))
        return lp;
      if(!check_finite(function, alpha, "Shape parameter", 
                       &lp, Policy()))
        return lp;
      if(!check_positive(function, alpha, "Shape parameter",
                         &lp, Policy()))
        return lp;
      if(!check_not_nan(function, sigma, "Scale parameter",
                        &lp, Policy()))
        return lp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &lp, Policy()))
        return lp;

      if (y < 0)
        return LOG_ZERO;
      
      using stan::math::multiply_log;
      
      if (include_summand<propto,T_shape>::value)
        lp += log(alpha);
      if (include_summand<propto,T_y,T_shape>::value)
        lp += multiply_log(alpha-1.0, y);
      if (include_summand<propto,T_shape,T_scale>::value)
        lp -= multiply_log(alpha, sigma);
      if (include_summand<propto,T_y,T_shape,T_scale>::value)
        lp -= pow(y / sigma, alpha);
      return lp;
    }


    template <bool propto,
              typename T_y, typename T_shape, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_log<propto>(y,alpha,sigma,stan::math::default_policy());
    }


    template <typename T_y, typename T_shape, typename T_scale, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma, 
                const Policy&) {
      return weibull_log<false>(y,alpha,sigma,Policy());
    }


    template <typename T_y, typename T_shape, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
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
