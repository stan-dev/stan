#ifndef __STAN__PROB__DISTRIBUTIONS__POISSON_HPP__
#define __STAN__PROB__DISTRIBUTIONS__POISSON_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>

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

      typename promote_args<T_rate>::type lp(0.0);
      if(!stan::prob::check_nonnegative(function, lambda, "Rate parameter, lambda,", &lp, Policy()))
	return lp;
      if(!stan::prob::check_nonnegative(function, n, "Number n", &lp, Policy()))
	return lp;
      
      if (lambda == 0)
	return LOG_ZERO;

      using stan::maths::multiply_log;

      if (include_summand<propto>::value)
	lp -= lgamma(n + 1.0);
      if (include_summand<propto,T_rate>::value)
	lp += multiply_log(n, lambda) - lambda;
      return lp;
    }
    


  }
}
#endif
