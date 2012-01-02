#ifndef __STAN__PROB__DISTRIBUTIONS__HYPERGEOMETRIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__HYPERGEOMETRIC_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;
    
    // Hypergeometric(n|N,a,b)  [0 <= n <= a;  0 <= N-n <= b;  0 <= N <= a+b]
    // n: #white balls drawn;  N: #balls drawn;  a: #white balls;  b: #black balls
    template <bool propto = false, 
	      class Policy = policy<> >
    inline double
    hypergeometric_log(const unsigned int n, const unsigned int N, 
		       const unsigned int a, const unsigned int b, const Policy& = Policy()) {
      // FIXME: domain checks
      return maths::binomial_coefficient_log(a,n)
	+ maths::binomial_coefficient_log(b,N-n)
	- maths::binomial_coefficient_log(a+b,N);
    }

  }
}
#endif
