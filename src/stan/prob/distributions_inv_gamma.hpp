#ifndef __STAN__PROB__DISTRIBUTIONS_INV_GAMMA_HPP__
#define __STAN__PROB__DISTRIBUTIONS_INV_GAMMA_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>


namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    /**
     * The log of an inverse gamma density for y with the specified
     * shape and scale parameters.
     * Shape and scale parameters must be greater than 0.
     * y must be greater than 0.
     * 
     \f{eqnarray*}{
       y &\sim& \mbox{\sf{Inv-gamma}}(\alpha, \beta) \\
       \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\beta^\alpha}{\Gamma(\alpha)} y^{-(\alpha + 1)} \exp^{- \beta / y} \right) \\
       &=& \alpha \log(\beta) - \log(\Gamma(\alpha)) - (\alpha + 1) \log(y) - \frac{\beta}{y} \\
       & & \mathrm{where } y > 0
     \f}
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_scale Type of scale.
     */
    template <bool propto = false,
	      typename T_y, typename T_shape, typename T_scale, 
	      class Policy = policy<> >
      inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
      inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta, const Policy& = Policy()) {
      static const char* function = "stan::prob::inv_gamma_log<%1%>(%1%)";

      double result;
      if (!stan::prob::check_positive(function, alpha, "Shape parameter", &result, Policy())) 
	return result;
      if (!stan::prob::check_positive(function, beta, "Scale parameter", &result, Policy())) 
	return result;
      if (!stan::prob::check_positive(function, y, "Random variate y", &result, Policy()))
	return result;
      
      typename promote_args<T_y,T_shape,T_scale>::type lp(0.0);
      if (!propto)
	lp -= lgamma(alpha);
      if (!propto 
	  || !is_constant<T_shape>::value
	  || !is_constant<T_scale>::value)
	lp += alpha * log(beta);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_shape>::value)
	lp -= (alpha + 1.0) * log(y);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_scale>::value)
	lp -= beta / y;
      return lp;
    }
      
    
  }
}

#endif
