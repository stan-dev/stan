#ifndef __STAN__PROB__DISTRIBUTIONS_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_BINOMIAL_HPP__

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

    // Binomial(n|N,theta)  [N >= 0;  0 <= n <= N;  0 <= theta <= 1]
    template <typename T_n, typename T_N, typename T_prob, class Policy>
    inline typename boost::math::tools::promote_args<T_prob>::type
    binomial_log(const T_n& n, const T_N& N, const T_prob& theta, const Policy& /* pol */) {
      return maths::binomial_coefficient_log<T_N>(N,n)
	+ n * log(theta)
	+ (N - n) * log(1.0 - theta);
    }

    template <typename T_n, typename T_N, typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    binomial_log(const T_n& n, const T_N& N, const T_prob& theta) {
      return binomial_log (n, N, theta, boost::math::policies::policy<>());
    }

    
    template <typename T_n, typename T_N, typename T_prob, class Policy>
    inline typename boost::math::tools::promote_args<T_prob>::type
    binomial_propto_log(const T_n& n, const T_N& N, const T_prob& theta, const Policy& /* pol */) {
      return binomial_log (n, N, theta, Policy());
    }

    template <typename T_n, typename T_N, typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    binomial_propto_log(const T_n& n, const T_N& N, const T_prob& theta) {
      return binomial_propto_log (n, N, theta, boost::math::policies::policy<>());
    }

  }
}
#endif
