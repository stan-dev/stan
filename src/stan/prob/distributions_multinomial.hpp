#ifndef __STAN__PROB__DISTRIBUTIONS_MULTINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_MULTINOMIAL_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {

  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    using Eigen::Matrix;
    using Eigen::Dynamic;

    // Multinomial(ns|N,theta)   [0 <= n <= N;  SUM ns = N;   
    //                            0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto = false,
	      typename T_prob, class Policy = policy<> >
    inline typename promote_args<T_prob>::type
    multinomial_log(const std::vector<int>& ns,
		    const Matrix<T_prob,Dynamic,1>& theta, 
		    const Policy& = Policy()) {
      // FIXME: domain checks
      typename promote_args<T_prob>::type lp(0.0);
      if (!propto) {	
	double sum = 1.0;
	for (unsigned int i = 0; i < ns.size(); ++i) 
	  sum += ns[i];
	lp += lgamma(sum);
	for (unsigned int i = 0; i < ns.size(); ++i)
	lp -= lgamma(ns[i] + 1.0);
      }
      if (!propto
	  || !is_constant<T_prob>::value)
	for (unsigned int i = 0; i < ns.size(); ++i)
	  lp += ns[i] * log(theta[i]);
      return lp;
    }


  }
}
#endif
