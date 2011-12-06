#ifndef __STAN__PROB__DISTRIBUTIONS_UNIFORM_HPP__
#define __STAN__PROB__DISTRIBUTIONS_UNIFORM_HPP__

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

    // CONTINUOUS, UNIVARIATE DENSITIES
    /**
     * The log of a uniform density for the given 
     * y, lower, and upper bound. 
     *
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{U}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta)) &=& \log \left( \frac{1}{\beta-\alpha} \right) \\
     &=& \log (1) - \log (\beta - \alpha) \\
     &=& -\log (\beta - \alpha) \\
     & & \mathrm{ where } \; y \in [\alpha, \beta], \log(0) \; \mathrm{otherwise}
     \f}
     * 
     * @param y A scalar variable.
     * @param alpha Lower bound.
     * @param beta Upper bound.
     * @throw std::invalid_argument if the lower bound is greater than 
     *    or equal to the lower bound
     * @tparam T_y Type of scalar.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_low, typename T_high, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta, const Policy& /* pol */) {
      static const char* function = "stan::prob::uniform_log<%1%>(%1%)";
      
      double result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;
      if(!stan::prob::check_bounds(function, alpha, beta, &result, Policy()))
	return result;
      
      if (y < alpha || y > beta)
	return LOG_ZERO;
      return -log(beta - alpha);
    }
     
    // CONTINUOUS, UNIVARIATE DENSITIES
    /**
     * The log of a uniform density for the given 
     * y, lower, and upper bound. 
     *
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{U}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta)) &=& \log \left( \frac{1}{\beta-\alpha} \right) \\
     &=& \log (1) - \log (\beta - \alpha) \\
     &=& -\log (\beta - \alpha) \\
     & & \mathrm{ where } \; y \in [\alpha, \beta], \log(0) \; \mathrm{otherwise}
     \f}
     * 
     * @param y A scalar variable.
     * @param alpha Lower bound.
     * @param beta Upper bound.
     * @throw std::invalid_argument if the lower bound is greater than 
     *    or equal to the lower bound
     * @tparam T_y Type of scalar.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_low, typename T_high>
    inline typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      return uniform_log (y, alpha, beta, boost::math::policies::policy<>());
    }


    /**
     * The log of a density proportional to a uniform density for the given 
     * y, lower, and upper bound. 
     *
     * @param y A scalar variable.
     * @param alpha Lower bound.
     * @param beta Upper bound.
     * @throw std::invalid_argument if the lower bound is greater than 
     *    or equal to the lower bound
     * @tparam T_y Type of scalar.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_low, typename T_high, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_propto_log(const T_y& y, const T_low& alpha, const T_high& beta, const Policy& /* pol */) {
      return uniform_log (y, alpha, beta, Policy());
    }

    /**
     * The log of a density proportional to a uniform density for the given 
     * y, lower, and upper bound. 
     *
     * @param y A scalar variable.
     * @param alpha Lower bound.
     * @param beta Upper bound.
     * @throw std::invalid_argument if the lower bound is greater than 
     *    or equal to the lower bound
     * @tparam T_y Type of scalar.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_low, typename T_high>
    inline typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_propto_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      return uniform_propto_log (y, alpha, beta, boost::math::policies::policy<>());
    }

  }}

#endif
