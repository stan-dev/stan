#ifndef __STAN__PROB__DISTRIBUTIONS_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_NORMAL_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {

  namespace prob {

    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    /**
     * The log of the normal density for the given y, mean, and
     * standard deviation.  The standard deviation must be greater
     * than 0.
     *
     * \f{eqnarray*}{
     y &\sim& \mbox{\sf{N}} (\mu, \sigma^2) \\
     \log (p (y \,|\, \mu, \sigma) ) &=& \log \left( \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( - \frac{1}{2 \sigma^2} (y - \mu)^2 \right) \right) \\
     &=& \log (1) - \frac{1}{2}\log (2 \pi) - \log (\sigma) - \frac{(y - \mu)^2}{2 \sigma^2}
     \f}
     * 
     * Errors are configured by policy.  All variables must be finite
     * and the scale must be strictly greater than zero.
     * 
     * @param y A scalar variate.
     * @param mu The location of the normal distribution.
     * @param sigma The scale of the normal distribution. 
     * @return The log of the normal density of the specified arguments.
     * @tparam propto Set to <code>true</code> if only calculated up to a proportion.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam Policy Error-handling policy.
     */
    template <bool propto = false, 
	      typename T_y, typename T_loc, typename T_scale, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_loc,T_scale>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
	       const Policy& = Policy()) {
      static const char* function = "stan::prob::normal_log<%1%>(%1%)";

      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if (!stan::prob::check_scale(function, sigma, &lp, Policy()))
	return lp;
      if (!stan::prob::check_location(function, mu, &lp, Policy()))
	return lp;
      if (!stan::prob::check_x(function, y, &lp, Policy()))
	return lp;
      
      if (!propto 
	  || !stan::is_constant<T_y>::value 
	  || !stan::is_constant<T_loc>::value 
	  || !stan::is_constant<T_scale>::value) 
	lp -= square(y - mu) / (2.0 * square(sigma));
      if (!propto
	  || !stan::is_constant<T_scale>::value)
	lp -= log(sigma);
      if (!propto)
	lp += NEG_LOG_SQRT_TWO_PI;
      
      return lp;
    }

    /**
     * Calculates the normal cumulative distribution function for the given
     * variate, location, and scale.
     * 
     * \f$\Phi(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\inf}^x e^{-t^2/2} dt\f$.
     * 
     * Errors are configured by policy.  All variables must be finite
     * and the scale must be strictly greater than zero.
     * 
     * @param y A scalar variate.
     * @param mu The location of the normal distribution.
     * @param sigma The scale of the normal distriubtion
     * @return The unit normal cdf evaluated at the specified arguments.
     * @tparam propto Set to <code>true</code> if only calculated up to a proportion.
     * @tparam T_y Type of y.
     * @tparam T_loc Type of mean parameter.
     * @tparam T_scale Type of standard deviation paramater.
     * @tparam Policy Error-handling policy.
     */
    template <bool propto = false, typename T_y, typename T_loc, typename T_scale, class Policy = policy<> >
    inline typename promote_args<T_y, T_loc, T_scale>::type
    normal_p(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */ = Policy() ) {
      static const char* function = "stan::prob::normal_p(%1%)";

      typename promote_args<T_y, T_loc, T_scale>::type lp;
      if (!stan::prob::check_scale(function, sigma, &lp, Policy()))
	return lp;
      if (!stan::prob::check_location(function, mu, &lp, Policy()))
	return lp;
      if (!stan::prob::check_x(function, y, &lp, Policy()))
	return lp;

      if (propto 
	  && stan::is_constant<T_y>::value 
	  && stan::is_constant<T_loc>::value 
	  && stan::is_constant<T_scale>::value)
	return 1.0;

      if (!propto)
	return 0.5 * erfc(-(y - mu)/(sigma * SQRT_2));
      
      return erfc(-(y - mu)/(sigma * SQRT_2));
    }


    /**
     * The log of the normal density for the specified sequence of
     * scalars given the specified mean and deviation.  If the
     * sequence of values is of length 0, the result is 0.0.
     *
     * <p>The result log probability is defined to be the sum of the
     * log probabilities for each observation.  Hence if the sequence
     * is of length 0, the log probability is 0.0.
     *
     * @param y Sequence of scalars.
     * @param mu Location parameter for the normal distribution.
     * @param sigma Scale parameter for the normal distribution.
     * @return The log of the product of the densities.
     * @throw std::domain_error if the scale is not positive.
     * @tparam T_y Underlying type of scalar in sequence.
     * @tparam T_loc Type of location parameter.
     */
    template <bool propto = false,
	      typename T_y, typename T_loc, typename T_scale, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_loc,T_scale>::type
    normal_log(const std::vector<T_y>& y,
	       const T_loc& mu,
	       const T_scale& sigma,
	       const Policy& /* pol */ = Policy()) {
      static const char* function = "stan::prob::normal_log<%1%>(%1%)";

      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if (!stan::prob::check_scale(function, sigma, &lp, Policy()))
	return lp;
      if (!stan::prob::check_location(function, mu, &lp, Policy()))
	return lp;
      if (!stan::prob::check_x(function, y, &lp, Policy()))
	return lp;

      if (y.size() == 0)
	return lp;
      
      if (!propto 
	  || !is_constant<T_y>::value 
	  || !is_constant<T_loc>::value
	  || !is_constant<T_scale>::value) {
	for (unsigned int n = 0; n < y.size(); ++n)
	  lp -= square(y[n] - mu);
	lp /= 2.0 * square(sigma);
      }
      
      if (!propto || !is_constant<T_scale>::value) 
	lp -= y.size() * log(sigma);
      
      if (!propto) 
	lp += y.size() * NEG_LOG_SQRT_TWO_PI;
      
      return lp;
    }


  }
}
#endif
