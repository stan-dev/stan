#ifndef __STAN__PROB__DISTRIBUTIONS_HYPERGEOMETRIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS_HYPERGEOMETRIC_HPP__

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/policies/policy.hpp>

#include "stan/prob/transform.hpp"
#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

namespace stan {
  namespace prob {
    using namespace std;
    using namespace stan::maths;


    // Hypergeometric(n|N,a,b)  [0 <= n <= a;  0 <= N-n <= b;  0 <= N <= a+b]
    // n: #white balls drawn;  N: #balls drawn;  a: #white balls;  b: #black balls
    template <class Policy>
    inline double
    hypergeometric_log(const unsigned int n, const unsigned int N, 
		       const unsigned int a, const unsigned int b, const Policy& /* pol */) {
      return maths::binomial_coefficient_log(a,n)
	+ maths::binomial_coefficient_log(b,N-n)
	- maths::binomial_coefficient_log(a+b,N);
    }
    
    inline double
    hypergeometric_log(const unsigned int n, const unsigned int N, 
		       const unsigned int a, const unsigned int b) {
      return hypergeometric_log (n, N, a, b, boost::math::policies::policy<>());
    }


    template <class Policy>
    inline double
    hypergeometric_propto_log(const unsigned int n, const unsigned int N, 
			      const unsigned int a, const unsigned int b, const Policy& /* pol */) {
      return hypergeometric_log (n, N, a, b, Policy());
    }
    
    inline double
    hypergeometric_propto_log(const unsigned int n, const unsigned int N, 
			      const unsigned int a, const unsigned int b) {
      return hypergeometric_propto_log (n, N, a, b, boost::math::policies::policy<>());
    }

  }
}
#endif
