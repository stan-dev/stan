#ifndef __STAN__PROB__DISTRIBUTIONS_POISSON_HPP__
#define __STAN__PROB__DISTRIBUTIONS_POISSON_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // Poisson(n|lambda)  [lambda > 0;  n >= 0]
    template <bool propto = false,
	      typename T_rate, 
	      class Policy = policy<> >
    inline typename promote_args<T_rate>::type
      poisson_log(const unsigned int n, const T_rate& lambda, const Policy& = Policy()) {
      static const char* function = "stan::prob::poisson_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_nonnegative(function, lambda, "Rate parameter, lambda,", &result, Policy()))
	return result;
      if(!stan::prob::check_nonnegative(function, n, "Number n", &result, Policy()))
	return result;
      

      if (lambda == 0)
	return LOG_ZERO;

      typename promote_args<T_rate>::type lp(0.0);
      if (!propto)
	lp -= lgamma (n + 1.0);
      if (!propto
	  || !is_constant<T_rate>::value)
	lp += n * log(lambda) - lambda;
      return lp;
    }
    


  }
}
#endif
