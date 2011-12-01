#ifndef __STAN__PROB__DISTRIBUTIONS_SCALED_INV_CHI_SQUARE_HPP__
#define __STAN__PROB__DISTRIBUTIONS_SCALED_INV_CHI_SQUARE_HPP__

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
     * The log of a scaled inverse chi-squared density for y with the specified
     * degrees of freedom parameter and scale parameter.
     * The degrees of freedom prarameter must be greater than 0. The scale parameter must be greater
     * than 0.
     * y must be greater than 0.
     * 
     \f{eqnarray*}{
       y &\sim& \mbox{\sf{Inv-}}\chi^2(\nu, s^2) \\
       \log (p (y \,|\, \nu, s)) &=& \log \left( \frac{(\nu / 2)^{\nu / 2}}{\Gamma (\nu / 2)} s^\nu y^{- (\nu / 2 + 1)} \exp^{-\nu s^2 / (2y)} \right) \\
       &=& \frac{\nu}{2} \log(\frac{\nu}{2}) - \log (\Gamma (\nu / 2)) + \nu \log(s) - (\frac{\nu}{2} + 1) \log(y) - \frac{\nu s^2}{2y} \\
       & & \mathrm{ where } \; y > 0
     \f}
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param s Scale parameter.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if s is not greater than or equal to 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s) {
      /*static const char* function = "stan::prob::scaled_inv_chi_square_log<%1%>(%1%)";
      
	typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type result;*/
      if (nu <= 0) {
	std::ostringstream err;
	err << "nu (" << nu << " must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      if (s <= 0) {
	std::ostringstream err;
	err << "s (" << s << " must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }      
      if (y <= 0) {
	std::ostringstream err;
	err << "y (" << y << ") must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      T_dof half_nu = 0.5 * nu;
      return - lgamma(half_nu)
	+ (half_nu) * log(half_nu)
	+ nu * log(s)
	- (half_nu + 1.0) * log(y)
	- half_nu * s * s / y;
    }
    /**
     * The log of a distribution proportional to a scaled inverse chi-squared density for y with the specified
     * degrees of freedom parameter and scale parameter.
     * The degrees of freedom prarameter must be greater than 0. The scale parameter must be greater
     * than 0.
     * y must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param s Scale parameter.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if s is not greater than or equal to 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_propto_log(const T_y& y, const T_dof& nu, const T_scale& s) {
      if (nu <= 0) {
	std::ostringstream err;
	err << "nu (" << nu << " must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      if (s <= 0) {
	std::ostringstream err;
	err << "s (" << s << " must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }      
      if (y <= 0) {
	std::ostringstream err;
	err << "y (" << y << ") must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      return scaled_inv_chi_square_log(y, nu, s);
    }
    /**
     * The log of a distribution proportional to a scaled inverse chi-squared density for y with the specified
     * degrees of freedom parameter and scale parameter.
     * The degrees of freedom prarameter must be greater than 0. The scale parameter must be greater
     * than 0.
     * y must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param s Scale parameter.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if s is not greater than or equal to 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_propto_log(stan::agrad::var& lp, const T_y& y, const T_dof& nu, const T_scale& s) {
      if (nu <= 0) {
	std::ostringstream err;
	err << "nu (" << nu << " must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      if (s <= 0) {
	std::ostringstream err;
	err << "s (" << s << " must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }      
      if (y <= 0) {
	std::ostringstream err;
	err << "y (" << y << ") must be greater than 0";
	BOOST_THROW_EXCEPTION(std::domain_error (err.str()));
      }
      lp += scaled_inv_chi_square_propto_log(y, nu, s);
    }


  }
}

#endif

