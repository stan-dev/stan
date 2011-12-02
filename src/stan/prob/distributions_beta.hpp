#ifndef __STAN__PROB__DISTRIBUTIONS_BETA_HPP__
#define __STAN__PROB__DISTRIBUTIONS_BETA_HPP__

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

    // Beta(y|alpha,beta)  [alpha > 0;  beta > 0;  0 <= y <= 1]
    /**
     * The log of a beta density for y with the specified
     * prior sample sizes.
     * Prior sample sizes, alpha and beta, must be greater than 0.
     * y must be between 0 and 1 inclusive.
     * 
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Beta}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} y^{\alpha - 1} (1-y)^{\beta - 1} \right) \\
     &=& \log (\Gamma(\alpha + \beta)) - \log (\Gamma (\alpha) - \log(\Gamma(\beta)) + (\alpha-1) \log(y) + (\beta-1) \log(1 - y) \\
     & & \mathrm{where} \; y \in [0, 1]
     \f}
     * @param y A scalar variable.
     * @param alpha Prior sample size.
     * @param beta Prior sample size.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_alpha Type of prior sample size for alpha.
     * @tparam T_beta Type of prior sample size for beta.
     */
    template <typename T_y, typename T_alpha, typename T_beta, class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_alpha,T_beta>::type
    beta_log(const T_y& y, const T_alpha& alpha, const T_beta& beta, const Policy& /* pol */ = Policy() ) {
      static const char* function = "stan::prob::beta_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_positive(function, alpha, "Prior sample size, alpha,", &result, Policy()))
	return result;
      if(!stan::prob::check_positive(function, beta, "Prior sample size, beta,", &result, Policy()))
	return result;
      if(!stan::prob::check_bounded_x(function, y, 0, 1, &result, Policy()))
	return result;

      return lgamma(alpha + beta)
	- lgamma(alpha)
	- lgamma(beta)
	+ (alpha - 1.0) * log(y)
	+ (beta - 1.0) * log(1.0 - y);
    }
    /**
     * The log of a distribution proportional to a beta density for y with the specified
     * prior sample sizes.
     * Prior sample sizes, alpha and beta, must be greater than 0.
     * y must be between 0 and 1 inclusive.
     *
     * @param y A scalar variable.
     * @param alpha Prior sample size.
     * @param beta Prior sample size.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_alpha Type of prior sample size for alpha.
     * @tparam T_beta Type of prior sample size for beta.
     */
    template <typename T_y, typename T_alpha, typename T_beta, class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_alpha,T_beta>::type
    beta_propto_log(const T_y& y, const T_alpha& alpha, const T_beta& beta, const Policy& /* pol */ = Policy() ) {
      return beta_log (y, alpha, beta, Policy());
    }


  }
}
#endif
