#ifndef __STAN__PROB__DISTRIBUTIONS_BERNOULLI_HPP__
#define __STAN__PROB__DISTRIBUTIONS_BERNOULLI_HPP__

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

    // Bernoulli(n|theta)   [0 <= n <= 1;   0 <= theta <= 1]
    template <typename T_prob, class Policy> 
    inline typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_log(const unsigned int n, const T_prob& theta, const Policy& /* pol */) {
      // FIXME: domain checks
      return log(n ? theta : (1.0 - theta));
    }

    template <typename T_prob> 
    inline typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_log(const unsigned int n, const T_prob& theta) {
      return bernoulli_log (n, theta, boost::math::policies::policy<>());
    }

    template <typename T_prob, class Policy> 
    inline typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_propto_log(const unsigned int n, const T_prob& theta, const Policy& /* pol */) {
      return bernoulli_log (n, theta, Policy());
    }

    template <typename T_prob> 
    inline typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_propto_log(const unsigned int n, const T_prob& theta) {
      return bernoulli_propto_log (n, theta, boost::math::policies::policy<>());
    }
    

  }
}
#endif
