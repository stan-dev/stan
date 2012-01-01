#ifndef __STAN__PROB__DISTRIBUTIONS_CAUCHY_HPP__
#define __STAN__PROB__DISTRIBUTIONS_CAUCHY_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // Cauchy(y|mu,sigma)  [sigma > 0]
    template <bool propto = false,
	      typename T_y, typename T_loc, typename T_scale, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& = Policy()) {
      static const char* function = "stan::prob::cauchy_log<%1%>(%1%)";

      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if(!stan::prob::check_scale(function, sigma, &lp, Policy()))
	return lp;
      if(!stan::prob::check_location(function, mu, &lp, Policy()))
	return lp;
      if(!stan::prob::check_x(function, y, &lp, Policy()))
	return lp;

      if (!propto)
	lp += NEG_LOG_PI;
      if (!propto
	  || !is_constant<T_scale>::value)
	lp -= log(sigma);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_loc>::value
	  || !is_constant<T_scale>::value)
	lp -= log1p(square ((y - mu) / sigma));
      return lp;
    }

  }
}
#endif
