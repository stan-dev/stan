#ifndef __STAN__PROB__DISTRIBUTIONS__LOGNORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__LOGNORMAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    // LogNormal(y|mu,sigma)  [y >= 0;  sigma > 0]
    template <bool propto = false,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& = Policy()) {
      static const char* function = "stan::prob::lognormal_log<%1%>(%1%)";

      using stan::maths::check_not_nan;
      using stan::maths::check_finite;
      using stan::maths::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_scale>::type lp;
      if (!check_not_nan(function, y, "Random variate, y,", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter, mu,", &lp, Policy()))
        return lp;
      if (!check_positive(function, sigma, "Scale parameter, sigma,", &lp, Policy()))
        return lp;
      
      if (y <= 0)
        return LOG_ZERO;
      
      using stan::maths::square;
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
    
  }
}
#endif
