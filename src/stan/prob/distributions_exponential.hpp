#ifndef __STAN__PROB__DISTRIBUTIONS_EXPONENTIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_EXPONENTIAL_HPP__

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

    /**
     * The log of an exponential density for y with the specified
     * inverse scale parameter.
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
       y &\sim& \mbox{\sf{Expon}}(\beta) \\
       \log (p (y \,|\, \beta) ) &=& \log \left( \beta \exp^{-\beta y} \right) \\
       &=& \log (\beta) - \beta y \\
       & & \mathrm{where} \; y > 0
     \f}
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_inv_scale, class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_inv_scale>::type
    exponential_log(const T_y& y, const T_inv_scale& beta, const Policy& /* pol */ = Policy()) {
      static const char* function = "stan::prob::normal_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_inv_scale(function, beta, &result, Policy()))
	return result;
      if(!stan::prob::check_nonnegative(function, y, "Random variate y", &result, Policy()))
	return result;

      return log(beta)
	- beta * y;
    }
    /**
     * The log of a distribution proportional to an exponential density for y with the specified
     * inverse scale parameter.
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_inv_scale, class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_inv_scale>::type
    exponential_propto_log(const T_y& y, const T_inv_scale& beta, const Policy& /* pol */ = Policy()) {
      return exponential_log (y, beta, Policy());
    }

  }
}

#endif
