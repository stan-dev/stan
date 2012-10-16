#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__STUDENT_T_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__STUDENT_T_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
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
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma, 
                  const Policy&) {
      static const char* function = "stan::prob::student_t_log(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(nu) 
            && stan::length(mu)
	    && stan::length(sigma)))
        return 0.0;

      typename return_type<T_y,T_dof,T_loc,T_scale>::type logp = 0.0;

      // validate args (here done over var, which should be OK)
      if (!check_not_nan(function, y, "Random variable", &logp, Policy()))
        return logp;
      if(!check_finite(function, nu, "Degrees of freedom parameter", &logp, Policy()))
        return logp;
      if(!check_positive(function, nu, "Degrees of freedom parameter", &logp, Policy()))
        return logp;
      if (!check_finite(function, mu, "Location parameter", 
                        &logp, Policy()))
        return logp;
      if (!check_finite(function, sigma, "Scale parameter", 
                        &logp, Policy()))
        return logp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &logp, Policy()))
        return logp;

      
      if (!(check_consistent_sizes(function,
                                   y,nu,mu,sigma,
				   "Random variable","Degrees of freedom parameter","Location parameter","Scale parameter",
                                   &logp, Policy())))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
	return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, nu, mu, sigma);

      using stan::math::square;
      using boost::math::lgamma;

      using std::log;

      for (size_t n = 0; n < N; n++) {
	if (include_summand<propto,T_dof>::value)
	  logp += lgamma( (nu_vec[n] + 1.0) / 2.0) - lgamma(nu_vec[n] / 2.0);
	if (include_summand<propto>::value)
	  logp += NEG_LOG_SQRT_PI;
	if (include_summand<propto,T_dof>::value)
	  logp -= 0.5 * log(nu_vec[n]);
	if (include_summand<propto,T_scale>::value)
	  logp -= log(sigma_vec[n]);
	if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
	  logp -= ((nu_vec[n] + 1.0) / 2.0) 
	    * log1p( square(((y_vec[n] - mu_vec[n]) / sigma_vec[n])) / nu_vec[n]);
      }
      return logp;
    }

    template <bool propto, 
              typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
      return student_t_log<propto>(y,nu,mu,sigma,stan::math::default_policy());
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale,
              class Policy>
    inline
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma, 
                  const Policy&) {
      return student_t_log<false>(y,nu,mu,sigma,Policy());
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
      return student_t_log<false>(y,nu,mu,sigma,stan::math::default_policy());
    }

    
  }
}
#endif
