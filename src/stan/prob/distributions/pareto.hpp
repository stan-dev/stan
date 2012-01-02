#ifndef __STAN__PROB__DISTRIBUTIONS__PARETO_HPP__
#define __STAN__PROB__DISTRIBUTIONS__PARETO_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>

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
	  
      using stan::maths::multiply_log;
      typename promote_args<T_y,T_scale,T_shape>::type lp(0.0);
      if (include_summand<propto,T_shape>::value)
	lp += log(alpha);
      if (include_summand<propto,T_scale,T_shape>::value)
	lp += multiply_log(alpha, y_min);
      if (include_summand<propto,T_y,T_shape>::value)
	lp -= multiply_log(alpha+1.0, y);
      return lp;
    }

  }
}
#endif
