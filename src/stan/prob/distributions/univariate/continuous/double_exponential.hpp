#ifndef __STAN__PROB__DISTRIBUTIONS__DOUBLE_EXPONENTIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__DOUBLE_EXPONENTIAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // DoubleExponential(y|mu,sigma)  [sigma > 0]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                           const Policy&) {
      static const char* function
        = "stan::prob::double_exponential_log(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if(!check_finite(function, y, "Random variable", &lp, Policy()))
        return lp;
      if(!check_finite(function, mu, "Location parameter", 
                       &lp, Policy()))
        return lp;
      if(!check_finite(function, sigma, "Scale parameter", 
                       &lp, Policy()))
        return lp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &lp, Policy()))
        return lp;

      if (include_summand<propto>::value)
        lp += NEG_LOG_TWO;
      if (include_summand<propto,T_scale>::value)
        lp -= log(sigma);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        lp -= fabs(y - mu) / sigma;
      return lp;
    }


    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, const T_loc& mu, 
                           const T_scale& sigma) {
      return double_exponential_log<propto>(y,mu,sigma,
                                            stan::math::default_policy());
    }


    template <typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                           const Policy&) {
      return double_exponential_log<false>(y,mu,sigma,Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, const T_loc& mu, 
                           const T_scale& sigma) {
      return double_exponential_log<false>(y,mu,sigma,
                                           stan::math::default_policy());
    }

    /** 
     * Calculates the double exponential cumulative density function.
     *
     * \f$ f(y|\mu,\sigma) = \begin{cases} \
           \frac{1}{2} \exp\left(\frac{y-\mu}{\sigma}\right), \mbox{if } y < \mu \\ 
           1 - \frac{1}{2} \exp\left(-\frac{y-\mu}{\sigma}\right), \mbox{if } y \ge \mu \
           \end{cases}\f$
     * 
     * @param y A scalar variate.
     * @param mu The location parameter.
     * @param sigma The scale parameter.
     * 
     * @return The cumulative density function.
     */
    template <typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    double_exponential_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                         const Policy&) {
      static const char* function
        = "stan::prob::double_exponential_cdf(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if(!check_finite(function, y, "Random variable", &lp, Policy()))
        return lp;
      if(!check_finite(function, mu, "Location parameter", 
                       &lp, Policy()))
        return lp;
      if(!check_finite(function, sigma, "Scale parameter", 
                       &lp, Policy()))
        return lp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &lp, Policy()))
        return lp;
      
      if (y < mu)
        return exp((y-mu)/sigma)/2;
      else
        return 1 - exp((mu-y)/sigma)/2;
    }
    
    template <typename T_y, typename T_loc, typename T_scale>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    double_exponential_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return double_exponential_cdf(y,mu,sigma,stan::math::default_policy());
    }
    
  }
}
#endif
