#ifndef __STAN__PROB__DISTRIBUTIONS_BETA_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_BETA_BINOMIAL_HPP__

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
    
    // BetaBinomial(n|alpha,beta) [alpha > 0;  beta > 0;  n >= 0]
    template <typename T_size, class Policy>
    inline typename boost::math::tools::promote_args<T_size>::type
    beta_binomial_log(const int n, const int N, const T_size& alpha, const T_size& beta, const Policy& /* pol */) {
      return maths::binomial_coefficient_log(N,n)
	+ maths::beta_log(n + alpha, N - n + beta)
	- maths::beta_log(alpha,beta);
    }

    template <typename T_size>
    inline typename boost::math::tools::promote_args<T_size>::type
    beta_binomial_log(const int n, const int N, const T_size& alpha, const T_size& beta) {
      return beta_binomial_log (n, N, alpha, beta, boost::math::policies::policy<>());
    }

    template <typename T_size, class Policy>
    inline typename boost::math::tools::promote_args<T_size>::type
    beta_binomial_propto_log(const int n, const int N, const T_size& alpha, const T_size& beta, const Policy& /* pol */) {
      return beta_binomial_log (n, N, alpha, beta, Policy());
    }

    template <typename T_size>
    inline typename boost::math::tools::promote_args<T_size>::type
    beta_binomial_propto_log(const int n, const int N, const T_size& alpha, const T_size& beta) {
      return beta_binomial_propto_log (n, N, alpha, beta, boost::math::policies::policy<>());
    }

  }
}
#endif
