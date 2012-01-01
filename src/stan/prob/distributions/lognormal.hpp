#ifndef __STAN__PROB__DISTRIBUTIONS_LOGNORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_LOGNORMAL_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

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
      if (!propto)
	lp += NEG_LOG_SQRT_TWO_PI;
      if (!propto
	  || !is_constant<T_scale>::value)
	lp -= log(sigma);
      if (!propto
	  || !is_constant<T_y>::value)
	lp -= log (y);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_loc>::value
	  || !is_constant<T_scale>::value)
	lp -= pow(log(y) - mu,2.0) / (2.0 * sigma * sigma);
      return lp;
    }
    
  }
}
#endif
