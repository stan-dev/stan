#ifndef __STAN__PROB__DISTRIBUTIONS_STUDENT_T_HPP__
#define __STAN__PROB__DISTRIBUTIONS_STUDENT_T_HPP__

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

    // StudentT(y|nu,mu,sigma)  [nu > 0;   sigma > 0]
    /**
     * The log of the Student-t density for the given y, nu, mean, and
     * scale parameter.  The scale parameter must be greater
     * than 0.
     *
     * \f{eqnarray*}{
       y &\sim& t_{\nu} (\mu, \sigma^2) \\
       \log (p (y \,|\, \nu, \mu, \sigma) ) &=& \log \left( \frac{\Gamma((\nu + 1) /2)}
           {\Gamma(\nu/2)\sqrt{\nu \pi} \sigma} \left( 1 + \frac{1}{\nu} (\frac{y - \mu}{\sigma})^2 \right)^{-(\nu + 1)/2} \right) \\
       &=& \log( \Gamma( (\nu+1)/2 )) - \log (\Gamma (\nu/2) - \frac{1}{2} \log(\nu \pi) - \log(\sigma)
           -\frac{\nu + 1}{2} \log (1 + \frac{1}{\nu} (\frac{y - \mu}{\sigma})^2)
     \f}
     * 
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param mu The mean of the Student-t distribution.
     * @param sigma The scale parameter of the Student-t distribution.
     * @return The log of the Student-t density at y.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::domain_error if nu is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, 
	      typename T_dof, 
	      typename T_loc, 
	      typename T_scale,
	      class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */ = Policy()) {
      static const char* function = "stan::prob::student_t_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_positive(function, nu, "Degrees of freedom", &result, Policy()))
	return result;
      if(!stan::prob::check_scale(function, sigma, &result, Policy()))
	return result;
      if(!stan::prob::check_location(function, mu, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;

      return lgamma((nu + 1.0) / 2.0)
	- lgamma(nu / 2.0)
	- 0.5 * log(nu)
	+ NEG_LOG_SQRT_PI
	- log(sigma)
	- ((nu + 1.0) / 2.0) * log(1.0 + (((y - mu) / sigma) * ((y - mu) / sigma)) / nu);
    }
    /**
     * The log of a density proportional to the Student-t density for the given y, nu,
     * mean, and scale parameter.  The scale parameter must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param mu The mean of the Student-t distribution.
     * @param sigma The scale parameter of the Student-t distribution.
     * @return The log of the Student-t density at y.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::domain_error if nu is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, 
	      typename T_dof, 
	      typename T_loc, 
	      typename T_scale,
	      class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    student_t_propto_log(const T_y& y, const T_dof& nu, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */ = Policy()) {
      return student_t_log (y, nu, mu, sigma);
    }

  }
}
#endif
