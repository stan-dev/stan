#ifndef __STAN__PROB__DISTRIBUTIONS_NEG_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_NEG_BINOMIAL_HPP__

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

    // NegBinomial(n|alpha,beta)  [alpha > 0;  beta > 0;  n >= 0]
    template <typename T_shape, typename T_inv_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_log(const int n, const T_shape& alpha, const T_inv_scale& beta, const Policy& /* pol */) {
      return maths::binomial_coefficient_log<T_shape>(n + alpha - 1.0, n)
	+ alpha * log(beta / (beta + 1.0))
	+ n * -log(beta + 1.0);
    }

    template <typename T_shape, typename T_inv_scale>
    inline typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_log(const int n, const T_shape& alpha, const T_inv_scale& beta) {
      return neg_binomial_log (n, alpha, beta, boost::math::policies::policy<>());
    }

    template <typename T_shape, typename T_inv_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_propto_log(const int n, const T_shape& alpha, const T_inv_scale& beta, const Policy& /* pol */) {
      return neg_binomial_propto_log (n, alpha, beta, Policy());
    }

    template <typename T_shape, typename T_inv_scale>
    inline typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_propto_log(const int n, const T_shape& alpha, const T_inv_scale& beta) {
      return neg_binomial_propto_log (n, alpha, beta, boost::math::policies::policy<>());
    }



  }
}
#endif
