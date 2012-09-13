#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__SCALED_INV_CHI_SQUARE_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__SCALED_INV_CHI_SQUARE_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>


namespace stan {

  namespace prob {

    /**
     * The log of a scaled inverse chi-squared density for y with the
     * specified degrees of freedom parameter and scale parameter.
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
    template <bool propto,
              typename T_y, typename T_dof, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s, 
                              const Policy&) {
      static const char* function 
        = "stan::prob::scaled_inv_chi_square_log(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_dof,T_scale>::type lp;
      if (!check_not_nan(function, y, "Random variable", &lp, Policy()))
        return lp;
      if (!check_finite(function, nu, "Degrees of freedom parameter", &lp, Policy()))
        return lp;
      if (!check_positive(function, nu, "Degrees of freedom parameter", &lp, Policy()))
        return lp;
      if (!check_finite(function, s, "Scale parameter", &lp, Policy()))
        return lp;
      if (!check_positive(function, s, "Scale parameter", &lp, Policy()))
        return lp;

      if (y <= 0)
        return LOG_ZERO;
      
      using stan::math::multiply_log;
      using stan::math::square;

      lp = 0.0;
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

    template <bool propto,
              typename T_y, typename T_dof, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s) {
      return scaled_inv_chi_square_log<propto>(y,nu,s,
                                               stan::math::default_policy());
    }

    template <typename T_y, typename T_dof, typename T_scale, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s, 
                              const Policy&) {
      return scaled_inv_chi_square_log<false>(y,nu,s,Policy());
    }

    template <typename T_y, typename T_dof, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s) {
      return scaled_inv_chi_square_log<false>(y,nu,s,
                                              stan::math::default_policy());
    }

  }
}

#endif

