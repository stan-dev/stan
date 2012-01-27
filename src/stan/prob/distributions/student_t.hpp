#ifndef __STAN__PROB__DISTRIBUTIONS__STUDENT_T_HPP__
#define __STAN__PROB__DISTRIBUTIONS__STUDENT_T_HPP__

#include <stan/prob/constants.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
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
    template <bool propto = false,
              typename T_y, 
              typename T_dof, 
              typename T_loc, 
              typename T_scale,
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, const T_scale& sigma, 
                  const Policy& = Policy()) {
      static const char* function = "stan::prob::student_t_log<%1%>(%1%)";

      using stan::maths::check_positive;
      using stan::maths::check_finite;
      using stan::maths::check_not_nan;
      using boost::math::tools::promote_args;
            
      typename promote_args<T_y,T_dof,T_loc,T_scale>::type lp;
      if(!check_positive(function, nu, "Degrees of freedom", &lp, Policy()))
        return lp;
      if (!check_positive(function, sigma, "Scale parameter, sigma,", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter, mu,", &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "Random variate y", &lp, Policy()))
        return lp;

      using stan::maths::square;
      using boost::math::lgamma;

      lp = 0.0;
      if (include_summand<propto,T_dof>::value)
        lp += lgamma( (nu + 1.0) / 2.0) - lgamma(nu / 2.0);
      if (include_summand<propto>::value)
        lp += NEG_LOG_SQRT_PI;
      if (include_summand<propto,T_dof>::value)
        lp -= 0.5 * log(nu);
      if (include_summand<propto,T_scale>::value)
        lp -= log(sigma);
      if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
        lp -= ((nu + 1.0) / 2.0) * log1p( square(((y - mu) / sigma)) / nu);
      return lp;
    }
    
  }
}
#endif
