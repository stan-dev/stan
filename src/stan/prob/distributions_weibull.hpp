#ifndef __STAN__PROB__DISTRIBUTIONS_WEIBULL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_WEIBULL_HPP__

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

    // Weibull(y|sigma,alpha)     [y >= 0;  sigma > 0;  alpha > 0]
    template <typename T_y, typename T_shape, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma, const Policy& /* pol */) {
      //static const char* function = "stan::prob::weibull_log<%1%>(%1%)";

      //double result;
      // FIXME: domain checks

      return log(alpha)
	- log(sigma)
	+ (alpha - 1.0) * (log(y) - log(sigma))
	- pow(y / sigma, alpha);
    }

    template <typename T_y, typename T_shape, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_log (y, alpha, sigma, boost::math::policies::policy<>());
    }

    template <typename T_y, typename T_shape, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_propto_log(const T_y& y, const T_shape& alpha, const T_scale& sigma, const Policy& /* pol */) {
      return weibull_log (y, alpha, sigma, Policy());
    }
    
    template <typename T_y, typename T_shape, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_propto_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_propto_log (y, alpha, sigma, boost::math::policies::policy<>());
    }

    template <typename T_y, typename T_shape, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_p(const T_y& y, const T_shape& alpha, const T_scale& sigma, const Policy& /* pol */) {
      //static const char* function = "stan::prob::weibull_p<%1%>(%1%)";

      //double result;
      // FIXME: domain checks
      
      if (y < 0)
	return 0;
      return 1.0 - exp (- pow (y / sigma, alpha));
    }

    template <typename T_y, typename T_shape, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    weibull_p(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return weibull_p (y, alpha, sigma, boost::math::policies::policy<>());
    }

    
  }
}
#endif
