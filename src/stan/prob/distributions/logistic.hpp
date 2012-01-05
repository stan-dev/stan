#ifndef __STAN__PROB__DISTRIBUTIONS__LOGISTIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__LOGISTIC_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>

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
