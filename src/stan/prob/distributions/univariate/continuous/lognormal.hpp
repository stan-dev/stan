#ifndef __STAN__PROB__DISTRIBUTIONS__LOGNORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__LOGNORMAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    // LogNormal(y|mu,sigma)  [y >= 0;  sigma > 0]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                  const Policy&) {
      static const char* function = "stan::prob::lognormal_log(%1%)";

      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_scale>::type lp;
      if (!check_not_nan(function, y, "Random variable", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter", 
                        &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter", 
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &lp, Policy()))
        return lp;
      
      if (y <= 0)
        return LOG_ZERO;
      
      using stan::math::square;
      using std::log;
      
      lp = 0.0;
      if (include_summand<propto>::value)
        lp += NEG_LOG_SQRT_TWO_PI;
      if (include_summand<propto,T_scale>::value)
        lp -= log(sigma);
      if (include_summand<propto,T_y>::value)
        lp -= log(y);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        lp -= square(log(y) - mu) / (2.0 * sigma * sigma);
      return lp;
    }

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    inline 
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return lognormal_log<propto>(y,mu,sigma,stan::math::default_policy());
    }

    template <typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    inline 
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                  const Policy&) {
      return lognormal_log<false>(y,mu,sigma,Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return lognormal_log<false>(y,mu,sigma,stan::math::default_policy());
    }




    template <typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                const Policy&) {
      static const char* function = "stan::prob::lognormal_cdf(%1%)";

      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_scale>::type lp;
      if (!check_not_nan(function, y, "Random variable", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter", &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter", 
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &lp, Policy()))
        return lp;

      return 0.5 * erfc(-(log(y) - mu)/(sigma * SQRT_2));
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return lognormal_cdf(y,mu,sigma,stan::math::default_policy());
    }
    
  }
}
#endif
