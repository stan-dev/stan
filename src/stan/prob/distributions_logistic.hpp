#ifndef __STAN__PROB__DISTRIBUTIONS_LOGISTIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS_LOGISTIC_HPP__

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

    // Logistic(y|mu,sigma)    [sigma > 0]
    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */) {
      return -(y - mu)/sigma
	- log(sigma)
	- 2.0 * log(1.0 + exp(-(y - mu)/sigma));
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return logistic_log (y, mu, sigma, boost::math::policies::policy<>());
    }

    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */) {
      return logistic_log (y, mu, sigma, Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return logistic_propto_log (y, mu, sigma, boost::math::policies::policy<>());
    }

    
  }
}
#endif
