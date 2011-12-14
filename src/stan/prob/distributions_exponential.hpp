#ifndef __STAN__PROB__DISTRIBUTIONS_EXPONENTIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_EXPONENTIAL_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>


namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

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
    template <bool propto = false,
	      typename T_y, typename T_inv_scale, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_inv_scale>::type
    exponential_log(const T_y& y, const T_inv_scale& beta, const Policy& = Policy()) {
      static const char* function = "stan::prob::exponential_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_inv_scale(function, beta, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;
      
      typename promote_args<T_y,T_inv_scale>::type lp(0.0);
      if (!propto)
	lp += log(beta);
      if (!propto
	  || !is_constant<T_y>::value
	  || !is_constant<T_inv_scale>::value)
	lp -= beta * y;
      return lp;
    }
    
    /**
     * Calculates the exponential cumulative distribution function for the given
     * y and beta.
     *
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <bool propto = false, 
	      typename T_y, typename T_inv_scale, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_inv_scale>::type
    exponential_p(const T_y& y, const T_inv_scale& beta, const Policy& = Policy()) {
      static const char* function = "stan::prob::exponential_p<%1%>(%1%)";

      double result;
      if(!stan::prob::check_inv_scale(function, beta, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;
      
      if (y < 0)
	return 0.0;
      
      if (!propto)
	return 1.0 - exp(-beta * y);
      return 1.0;
    }


  }
}

#endif
