#ifndef __STAN__PROB__DISTRIBUTIONS_BETA_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_BETA_BINOMIAL_HPP__

#include <stan/meta/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>
#include <stan/maths/special_functions.hpp>

namespace stan {
  namespace prob {

    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // BetaBinomial(n|alpha,beta) [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto = false,
	      typename T_size, 
	      class Policy = policy<> >
      inline typename promote_args<T_size>::type
      beta_binomial_log(const int n, const int N, const T_size& alpha, const T_size& beta,
			const Policy& = Policy()) {

      using stan::maths::beta_log;
      using stan::maths::binomial_coefficient_log;

      // FIXME: domain checks
      typename promote_args<T_size>::type lp(0.0);
      if (!propto)
	lp += binomial_coefficient_log(N,n);
      if (!propto
	  || !is_constant<T_size>::value)
	lp += beta_log(n + alpha, N - n + beta) - beta_log(alpha,beta);
      return lp;
    }

  }
}
#endif
