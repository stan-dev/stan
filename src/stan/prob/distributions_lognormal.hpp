#ifndef __STAN__PROB__DISTRIBUTIONS_LOGNORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_LOGNORMAL_HPP__

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

    // LogNormal(y|mu,sigma)  [y >= 0;  sigma > 0]
    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */) {
      return NEG_LOG_SQRT_TWO_PI
	- log(sigma)
	- log(y)
	- pow(log(y) - mu,2.0) / (2.0 * sigma * sigma);
    }
    
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return lognormal_log (y, mu, sigma, boost::math::policies::policy<>());
    }

    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */) {
      return lognormal_log (y, mu, sigma, Policy());
    }
    
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    lognormal_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return lognormal_propto_log (y, mu, sigma, boost::math::policies::policy<>());
    }


  }
}
#endif
