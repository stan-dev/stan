#ifndef __STAN__PROB__DISTRIBUTIONS_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_BINOMIAL_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // Binomial(n|N,theta)  [N >= 0;  0 <= n <= N;  0 <= theta <= 1]
    template <bool propto = false,
	      typename T_n, typename T_N, typename T_prob, 
	      class Policy = policy<> >
    inline typename promote_args<T_prob>::type
      binomial_log(const T_n& n, const T_N& N, const T_prob& theta, const Policy& = Policy()) {
      // FIXME: domain checks
      typename promote_args<T_prob>::type lp(0.0);
      if (!propto)
	lp += maths::binomial_coefficient_log<T_N>(N,n);
      if (!propto
	  || !is_constant<T_n>::value
	  || !is_constant<T_N>::value
	  || !is_constant<T_prob>::value)
	lp += n * log(theta) + (N-n) * log1m (theta);
      return lp;
    }
  }
}
#endif
