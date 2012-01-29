#ifndef __STAN__PROB__DISTRIBUTIONS__LOGISTIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__LOGISTIC_HPP__

#include <stan/prob/constants.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    // Logistic(y|mu,sigma)    [sigma > 0]
    template <bool propto = false,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& = Policy()) {
      static const char* function = "stan::prob::logistic_log<%1%>(%1%)";
      
      using stan::maths::check_positive;
      using stan::maths::check_finite;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if (!check_positive(function, sigma, "Scale parameter, sigma,", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter, mu,", &lp, Policy()))
        return lp;
      if (!check_finite(function, y, "Random variate y", &lp, Policy()))
        return lp;

      using stan::maths::log1p;

      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        lp -= (y - mu)/sigma;
      if (include_summand<propto,T_scale>::value)
        lp -= log(sigma);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        lp -= 2.0 * log1p(exp(-(y - mu)/sigma));
      return lp;
    }
  }
}
#endif
