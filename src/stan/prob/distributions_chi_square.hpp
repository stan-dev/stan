#ifndef __STAN__PROB__DISTRIBUTIONS_CHI_SQUARE_HPP__
#define __STAN__PROB__DISTRIBUTIONS_CHI_SQUARE_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    /**
     * The log of a chi-squared density for y with the specified
     * degrees of freedom parameter.
     * The degrees of freedom prarameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
       y &\sim& \chi^2_\nu \\
       \log (p (y \,|\, \nu)) &=& \log \left( \frac{2^{-\nu / 2}}{\Gamma (\nu / 2)} y^{\nu / 2 - 1} \exp^{- y / 2} \right) \\
       &=& - \frac{\nu}{2} \log(2) - \log (\Gamma (\nu / 2)) + (\frac{\nu}{2} - 1) \log(y) - \frac{y}{2} \\
       & & \mathrm{ where } \; y \ge 0
     \f}
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <bool propto = false, 
	      typename T_y, typename T_dof, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_dof>::type
    chi_square_log(const T_y& y, const T_dof& nu, const Policy& = Policy()) {
      static const char* function = "stan::prob::chi_square_log<%1%>(%1%)";

      double result;
      if (!stan::prob::check_positive (function, nu, "Degrees of freedom", &result, Policy()))
	return result;
      if (!stan::prob::check_x (function, y, &result, Policy()))
	return result;
      
      if (y < 0)
	return LOG_ZERO;
      
      typename promote_args<T_y,T_dof>::type lp(0.0);
      
      if (!propto)
	lp += nu * NEG_LOG_TWO_OVER_TWO - lgamma(0.5 * nu);
      if (!propto
	  || !stan::is_constant<T_y>::value
	  || !stan::is_constant<T_dof>::value)
	lp += (0.5 * nu - 1.0) * log(y);
      if (!propto
	  || !stan::is_constant<T_y>::value)
	lp -= 0.5 * y;
      return lp;
    }

  }
}

#endif

