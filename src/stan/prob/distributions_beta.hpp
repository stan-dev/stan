#ifndef __STAN__PROB__DISTRIBUTIONS_BETA_HPP__
#define __STAN__PROB__DISTRIBUTIONS_BETA_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

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
    template <bool propto = false,
	      typename T_y, typename T_alpha, typename T_beta, 
	      class Policy = policy<> >
      inline typename promote_args<T_y,T_alpha,T_beta>::type
      beta_log(const T_y& y, const T_alpha& alpha, const T_beta& beta, const Policy& = Policy()) {
      static const char* function = "stan::prob::beta_log<%1%>(%1%)";

      typename promote_args<T_y,T_alpha,T_beta>::type lp(0.0);
      if(!stan::prob::check_positive(function, alpha, "Prior sample size, alpha,", &lp, Policy()))
	return lp;
      if(!stan::prob::check_positive(function, beta, "Prior sample size, beta,", &lp, Policy()))
	return lp;
      if(!stan::prob::check_bounded_x(function, y, 0, 1, &lp, Policy()))
	return lp;

      if (!propto 
	  || !is_constant<T_alpha>::value
	  || !is_constant<T_beta>::value)
	lp += lgamma(alpha + beta);
      if (!propto 
	  || !is_constant<T_alpha>::value)
      lp -= lgamma(alpha);
      if (!propto 
	  || !is_constant<T_beta>::value)
      lp -= lgamma(beta);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_alpha>::value)
	lp += (alpha - 1.0) * log(y);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_beta>::value)
	lp += (beta - 1.0) * log1m(y);
      return lp;
    }

  }
}
#endif
