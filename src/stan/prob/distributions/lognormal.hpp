#ifndef __STAN__PROB__DISTRIBUTIONS__LOGNORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__LOGNORMAL_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>


namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // LogNormal(y|mu,sigma)  [y >= 0;  sigma > 0]
    template <bool propto = false,
	      typename T_y, typename T_loc, typename T_scale, 
	      class Policy = policy<> >
      inline typename promote_args<T_y,T_loc,T_scale>::type
      lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& = Policy()) {
      //FIXME: bounds checks
      
      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if (include_summand<propto>::value)
	lp += NEG_LOG_SQRT_TWO_PI;
      if (include_summand<propto,T_scale>::value)
	lp -= log(sigma);
      if (include_summand<propto,T_y>::value)
	lp -= log(y);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
	lp -= pow(log(y) - mu,2.0) / (2.0 * sigma * sigma);
      return lp;
    }
    
  }
}
#endif
