#ifndef __STAN__PROB__DISTRIBUTIONS__CAUCHY_HPP__
#define __STAN__PROB__DISTRIBUTIONS__CAUCHY_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    using stan::maths::square;

    // Cauchy(y|mu,sigma)  [sigma > 0]
    template <bool propto = false,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy = policy<> >
    inline typename promote_args<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& = Policy()) {
      static const char* function = "stan::prob::cauchy_log<%1%>(%1%)";

      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if(!stan::prob::check_positive(function, sigma, "Scale parameter, sigma,", &lp, Policy()))
        return lp;
      if(!stan::prob::check_finite(function, mu, "Location parameter, mu,", &lp, Policy()))
        return lp;
      if(!stan::prob::check_not_nan(function, y, "Random variate, y,", &lp, Policy()))
        return lp;

      using stan::maths::log1p;
      
      if (include_summand<propto>::value)
        lp += NEG_LOG_PI;
      if (include_summand<propto,T_scale>::value)
        lp -= log(sigma);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        lp -= log1p(square ((y - mu) / sigma));
      return lp;
    }

  }
}
#endif
