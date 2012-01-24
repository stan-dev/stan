#ifndef __STAN__PROB__DISTRIBUTIONS__SCALED_INV_CHI_SQUARE_HPP__
#define __STAN__PROB__DISTRIBUTIONS__SCALED_INV_CHI_SQUARE_HPP__

#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/traits.hpp>


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
     * @throw std::domain_error if nu is not greater than 0
     * @throw std::domain_error if s is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <bool propto = false,
              typename T_y, typename T_dof, typename T_scale, 
              class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s, 
                              const Policy& = Policy()) {
      static const char* function = "stan::prob::scaled_inv_chi_square_log<%1%>(%1%)";
      
      using stan::maths::multiply_log;
      using stan::maths::square;

      typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type lp(0.0);
      if (!check_positive(function, nu, "Degrees of freedom", &lp, Policy()))
        return lp;
      if (!check_positive(function, s, "Scale", &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "Random variate y", &lp, Policy()))
        return lp;

      if (y <= 0)
        return LOG_ZERO;
      
      if (include_summand<propto,T_dof>::value) {
        T_dof half_nu = 0.5 * nu;
        lp += multiply_log(half_nu,half_nu) - lgamma(half_nu);
      }
      if (include_summand<propto,T_dof,T_scale>::value)
        lp += nu * log(s);
      if (include_summand<propto,T_dof,T_y>::value)
        lp -= multiply_log(nu*0.5+1.0, y);
      if (include_summand<propto,T_dof,T_y,T_scale>::value)
        lp -= nu * 0.5 * square(s) / y;
      return lp;
    }

  }
}

#endif

