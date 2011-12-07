#ifndef __STAN__PROB__DISTRIBUTIONS_PARETO_HPP__
#define __STAN__PROB__DISTRIBUTIONS_PARETO_HPP__

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

    // Pareto(y|y_m,alpha)  [y > y_m;  y_m > 0;  alpha > 0]
    template <typename T_y, typename T_scale, typename T_shape, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha, const Policy& /* pol */) {
      // FIXME: bounds checks
      return log(alpha)
	+ alpha * log(y_min)
	- (alpha + 1.0) * log(y);
    }

    template <typename T_y, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
      return pareto_log (y, y_min, alpha, boost::math::policies::policy<>());
    }

    template <typename T_y, typename T_scale, typename T_shape, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_propto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha, const Policy& /* pol */) {
      return pareto_log (y, y_min, alpha, Policy());
    }

    template <typename T_y, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_propto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
      return pareto_propto_log (y, y_min, alpha, boost::math::policies::policy<>());
    }
    

  }
}
#endif
