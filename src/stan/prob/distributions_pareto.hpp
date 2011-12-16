#ifndef __STAN__PROB__DISTRIBUTIONS_PARETO_HPP__
#define __STAN__PROB__DISTRIBUTIONS_PARETO_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // Pareto(y|y_m,alpha)  [y > y_m;  y_m > 0;  alpha > 0]
    template <bool propto = false,
	      typename T_y, typename T_scale, typename T_shape, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_scale,T_shape>::type
      pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha, const Policy& = Policy()) {
      // FIXME: bounds checks
      if (y < y_min)
	return LOG_ZERO;
	  
      typename promote_args<T_y,T_scale,T_shape>::type lp(0.0);
      if (!propto
	  || !is_constant<T_shape>::value)
	lp += log(alpha);
      if (!propto
	  || !is_constant<T_scale>::value
	  || !is_constant<T_shape>::value)
	lp += alpha * log(y_min);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_shape>::value)
	lp -= (alpha + 1.0) * log(y);
      return lp;
    }

  }
}
#endif
