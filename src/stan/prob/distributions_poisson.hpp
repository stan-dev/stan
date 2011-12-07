#ifndef __STAN__PROB__DISTRIBUTIONS_POISSON_HPP__
#define __STAN__PROB__DISTRIBUTIONS_POISSON_HPP__

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

    // Poisson(n|lambda)  [lambda > 0;  n >= 0]
    template <typename T_rate, class Policy>
    inline typename boost::math::tools::promote_args<T_rate>::type
    poisson_log(const unsigned int n, const T_rate& lambda, const Policy& /* pol */) {
      return - lgamma(n + 1.0)
	+ n * log(lambda)
	- lambda;
    }
    
    template <typename T_rate>
    inline typename boost::math::tools::promote_args<T_rate>::type
    poisson_log(const unsigned int n, const T_rate& lambda) {
      return poisson_log (n, lambda, boost::math::policies::policy<>());
    }

    template <typename T_rate, class Policy>
    inline typename boost::math::tools::promote_args<T_rate>::type
    poisson_propto_log(const unsigned int n, const T_rate& lambda, const Policy& /* pol */) {
      return poisson_log (n, lambda, Policy());
    }
    
    template <typename T_rate>
    inline typename boost::math::tools::promote_args<T_rate>::type
    poisson_propto_log(const unsigned int n, const T_rate& lambda) {
      return poisson_propto_log (n, lambda, boost::math::policies::policy<>());
    }


  }
}
#endif
