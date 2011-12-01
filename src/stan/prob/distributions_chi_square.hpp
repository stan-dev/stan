#ifndef __STAN__PROB__DISTRIBUTIONS_CHI_SQUARE_HPP__
#define __STAN__PROB__DISTRIBUTIONS_CHI_SQUARE_HPP__

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
    template <typename T_y, typename T_dof, class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_dof>::type
    chi_square_log(const T_y& y, const T_dof& nu, const Policy& /* pol */ = Policy()) {
      static const char* function = "stan::prob::chi_square_log<%1%>(%1%)";

      typename boost::math::tools::promote_args<T_y,T_dof>::type result;
      if (false == stan::prob::check_positive (function, nu, "Degrees of freedom", &result, Policy()))
	return result;
      if (false == stan::prob::check_nonnegative (function, y, "Random variate y", &result, Policy()))
	return result;
      return - lgamma(0.5 * nu)
	+ nu * NEG_LOG_TWO_OVER_TWO
	+ (0.5 * nu - 1.0) * log(y)
	- 0.5 * y;
    }

    /**
     * The log of a distribution proportional to a chi-squared density for y with the specified
     * degrees of freedom parameter.
     * The degrees of freedom prarameter must be greater than 0.
     * y must be greater than or equal to 0.
     *
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <typename T_y, typename T_dof, class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_dof>::type
    chi_square_propto_log(const T_y& y, const T_dof& nu, const Policy& /* pol */ = Policy()) {
      return chi_square_log(y, nu, Policy());
    } 

  }
}

#endif

