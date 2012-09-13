#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__STUDENT_T_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__STUDENT_T_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

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
    template <bool propto, typename T_y, typename T_dof, 
              typename T_loc, typename T_scale,
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma, 
                  const Policy&) {
      static const char* function = "stan::prob::student_t_log(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
            
      typename promote_args<T_y,T_dof,T_loc,T_scale>::type lp = 0.0;
      if (!check_not_nan(function, y, "Random variable", &lp, Policy()))
        return lp;
      if(!check_finite(function, nu, "Degrees of freedom parameter", &lp, Policy()))
        return lp;
      if(!check_positive(function, nu, "Degrees of freedom parameter", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter", 
                        &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter", 
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &lp, Policy()))
        return lp;

      using stan::math::square;
      using boost::math::lgamma;

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

    template <bool propto, 
              typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
      return student_t_log<propto>(y,nu,mu,sigma,stan::math::default_policy());
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale,
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma, 
                  const Policy&) {
      return student_t_log<false>(y,nu,mu,sigma,Policy());
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
      return student_t_log<false>(y,nu,mu,sigma,stan::math::default_policy());
    }

    
  }
}
#endif
