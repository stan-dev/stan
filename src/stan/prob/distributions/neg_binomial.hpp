#ifndef __STAN__PROB__DISTRIBUTIONS__NEG_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__NEG_BINOMIAL_HPP__

#include <stan/prob/traits.hpp>
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

      using stan::maths::multiply_log;

      typename promote_args<T_shape, T_inv_scale>::type lp(0.0);
      if (include_summand<propto>::value)
	lp += maths::binomial_coefficient_log<T_shape>(n + alpha - 1.0, n);
      if (include_summand<propto,T_shape,T_inv_scale>::value)
	lp += multiply_log(alpha, beta) - (alpha + n) * log1p(beta);
      return lp;
    }

  }
}
#endif
