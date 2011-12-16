#ifndef __STAN__PROB__DISTRIBUTIONS_LOGISTIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS_LOGISTIC_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // Logistic(y|mu,sigma)    [sigma > 0]
    template <bool propto = false,
	      typename T_y, typename T_loc, typename T_scale, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_loc,T_scale>::type
      logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& = Policy()) {
      // FIXME: bounds checks
      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_loc>::value
	  || !is_constant<T_scale>::value)
	lp -= (y - mu)/sigma;
      if (!propto
	  || !is_constant<T_scale>::value)
	lp -= log(sigma);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_loc>::value
	  || !is_constant<T_scale>::value)
	lp -= 2.0 * log(1.0 + exp(-(y - mu)/sigma));
      return lp;
    }


  }
}
#endif
