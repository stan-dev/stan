#ifndef __STAN__PROB__DISTRIBUTIONS_NEG_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_NEG_BINOMIAL_HPP__

#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // NegBinomial(n|alpha,beta)  [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto = false,
	      typename T_shape, typename T_inv_scale, 
	      class Policy = policy<> >
    inline typename promote_args<T_shape, T_inv_scale>::type
      neg_binomial_log(const int n, const T_shape& alpha, const T_inv_scale& beta, const Policy& = Policy()) {
      // FIXME: domain checks
      typename promote_args<T_shape, T_inv_scale>::type lp(0.0);
      if (!propto)
	lp += maths::binomial_coefficient_log<T_shape>(n + alpha - 1.0, n);
      if (!propto
	  || !is_constant<T_shape>::value
	  || !is_constant<T_inv_scale>::value)
	lp += alpha * log(beta) - (alpha + n) * log1p(beta);
      return lp;
    }

  }
}
#endif
