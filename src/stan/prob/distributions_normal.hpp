#ifndef __STAN__PROB__DISTRIBUTIONS_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_NORMAL_HPP__

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
    using namespace std;
    using namespace stan::maths;

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
    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_log(const std::vector<T_y>& y,
	       const T_loc& mu,
	       const T_scale& sigma,
	       const Policy& /* pol */) {
      static const char* function = "stan::prob::normal_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_scale(function, sigma, &result, Policy()))
	return result;
      if(!stan::prob::check_location(function, mu, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;

      double size = y.size();
      typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      for (unsigned int n = 0; n < y.size(); ++n)
	lp += square(y[n] - mu);
      return (size * NEG_LOG_SQRT_TWO_PI)
	- (lp / (2.0 * square(sigma)))
	+ (-size) * log(sigma);
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
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_log(const std::vector<T_y>& y,
	       const T_loc& mu,
	       const T_scale& sigma) {
      return normal_log (y, mu, sigma, boost::math::policies::policy<>() );
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
    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_propto_log(const std::vector<T_y>& y,
		      const T_loc& mu,
		      const T_scale& sigma,
		      const Policy& /* pol */) {
      return normal_log (y, mu, sigma, Policy());
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
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_propto_log(const std::vector<T_y>& y,
		      const T_loc& mu,
		      const T_scale& sigma) {
      return normal_propto_log (y, mu, sigma, boost::math::policies::policy<> ());
    }      

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
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @return The log of the normal density.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam Policy policy as defined by boost::math::policies
     */
    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */) {
      static const char* function = "stan::prob::normal_log<%1%>(%1%)";
      
      double result;
      if(!stan::prob::check_scale(function, sigma, &result, Policy()))
	return result;
      if(!stan::prob::check_location(function, mu, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;
      
      return (NEG_LOG_SQRT_TWO_PI
	      - log(sigma)
	      - square(y - mu) / (2.0 * square(sigma)));
    }

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
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @return The log of the normal density.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam Policy policy as defined by boost::math::policies
     */
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return normal_log (y, mu, sigma, boost::math::policies::policy<>());
    }



    /**
     * The log of the normal density up to a proportion for the given 
     * y, mean, and standard deviation.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @return The log of the normal density up to a proportion.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */    
    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */) {
      return normal_log(y,mu,sigma, Policy());
    }

    /**
     * The log of the normal density up to a proportion for the given 
     * y, mean, and standard deviation.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @return The log of the normal density up to a proportion.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */    
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    normal_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return normal_propto_log(y,mu,sigma, boost::math::policies::policy<>());
    }


    /**
     * Calculates the normal cumulative distribution function for the given
     * y, mean, and variance.
     * 
     * \f$\Phi(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\inf}^x e^{-t^2/2} dt\f$.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distriubtion
     * @return The unit normal cdf evaluated at the specified argument.
     * @throw std::domain_error if sigma is less than 0
     * @tparam T_y Type of y.
     * @tparam T_loc Type of mean parameter.
     * @tparam T_scale Type of standard deviation paramater.
     */
    template <typename T_y, typename T_loc, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale>::type
    normal_p(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& /* pol */) {
      static const char* function = "stan::prob::normal_p<%1%>(%1%)";

      double result;
      if(!stan::prob::check_scale(function, sigma, &result, Policy()))
	return result;
      if(!stan::prob::check_location(function, mu, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;

      return 0.5 * erfc(-(y - mu)/(sigma * SQRT_2));
    }
    
    /**
     * Calculates the normal cumulative distribution function for the given
     * y, mean, and variance.
     * 
     * \f$\Phi(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\inf}^x e^{-t^2/2} dt\f$.
     * 
     * @param y A scalar variable.
     * @param mean The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distriubtion
     * @return The unit normal cdf evaluated at the specified argument.
     * @throw std::domain_error if sigma is less than 0
     * @tparam T_y Type of y.
     * @tparam T_loc Type of mean parameter.
     * @tparam T_scale Type of standard deviation paramater.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale>::type
    normal_p(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return normal_p (y, mu, sigma, boost::math::policies::policy<>());
    }

    // NormalTruncatedLH(y|mu,sigma,low,high)  [sigma > 0, low < high]
    // Norm(y|mu,sigma) / (Norm_p(high|mu,sigma) - Norm_p(low|mu,sigma))
    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, lower bound, and upper bound.
     * The standard deviation must be greater than 0.
     * The lower bound must be less than the upper bound.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::invalid_argument if high is not greater than low.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low, typename T_high, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low, T_high>::type
    normal_trunc_lh_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low, const T_high& high, const Policy& /* pol */) {
      static const char* function = "stan::prob::normal_trunc_lh_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_scale(function, sigma, &result, Policy()))
	return result;
      if(!stan::prob::check_location(function, mu, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;
      if(!stan::prob::check_bounds(function, low, high, &result, Policy()))
	return result;

      if (y > high || y < low)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log(normal_p(high,mu,sigma) - normal_p(low,mu,sigma));
    }

    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, lower bound, and upper bound.
     * The standard deviation must be greater than 0.
     * The lower bound must be less than the upper bound.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::invalid_argument if high is not greater than low.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low, typename T_high>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low, T_high>::type
    normal_trunc_lh_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low, const T_high& high) {
      return normal_trunc_lh_log(y, mu, sigma, low, high, boost::math::policies::policy<>());
    }

    /**
     * The log of a distribution proportional to a truncated normal density for the given 
     * y, mean, standard deviation, lower bound, and upper bound.
     * The standard deviation must be greater than 0.
     * The lower bound must be less than the upper bound.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::invalid_argument if high is not greater than low.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low, typename T_high, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low, T_high>::type
    normal_trunc_lh_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low, const T_high& high, const Policy& /* pol */) {
      return normal_trunc_lh_log (y, mu, sigma, low, high, Policy());
    }
    /**
     * The log of a distribution proportional to a truncated normal density for the given 
     * y, mean, standard deviation, lower bound, and upper bound.
     * The standard deviation must be greater than 0.
     * The lower bound must be less than the upper bound.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::invalid_argument if high is not greater than low.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low, typename T_high>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low, T_high>::type
    normal_trunc_lh_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low, const T_high& high) {
      return normal_trunc_lh_propto_log (y, mu, sigma, low, high, boost::math::policies::policy<>());
    }

    // NormalTruncatedL(y|mu,sigma,low)  [sigma > 0]
    // Norm(y|mu,sigma) / (1 - Norm_p(low|mu,sigma))
    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and lower bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low>::type
    normal_trunc_l_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low, const Policy& /* pol */) {
      static const char* function = "stan::prob::normal_trunc_l_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_scale(function, sigma, &result, Policy()))
	return result;
      if(!stan::prob::check_location(function, mu, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;
      if(!stan::prob::check_lower_bound(function, low, &result, Policy()))
	return result;

      if (y < low)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log1m(normal_p(low,mu,sigma));
    }
    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and lower bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low>::type
    normal_trunc_l_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low) {
      return normal_trunc_l_log (y, mu, sigma, low, boost::math::policies::policy<>());
    }

    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and lower bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low>::type
    normal_trunc_l_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low, const Policy& /* pol */) {
      return normal_trunc_l_log (y, mu, sigma, low, Policy());
    }

    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and lower bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param low Lower bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_low Type of lower bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_low>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_low>::type
    normal_trunc_l_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_low& low) {
      return normal_trunc_l_propto_log (y, mu, sigma, low, boost::math::policies::policy<>());
    }

    // NormalTruncatedH(y|mu,sigma,high)  [sigma > 0]
    // Norm(y|mu,sigma) / (Norm_p(high|mu,sigma) - 0)
    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and upper bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_high, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_high>::type
    normal_trunc_h_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_high& high, const Policy& /* pol */) {
      static const char* function = "stan::prob::normal_trunc_h_log<%1%>(%1%)";

      double result;
      if(!stan::prob::check_scale(function, sigma, &result, Policy()))
	return result;
      if(!stan::prob::check_location(function, mu, &result, Policy()))
	return result;
      if(!stan::prob::check_x(function, y, &result, Policy()))
	return result;
      if(!stan::prob::check_upper_bound(function, high, &result, Policy()))
	return result;

      if (y > high)
	return LOG_ZERO;
      return normal_log(y,mu,sigma) 
	- log(normal_p(high,mu,sigma));
    }

    /**
     * The log of a truncated normal density for the given 
     * y, mean, standard deviation, and upper bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_high>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_high>::type
    normal_trunc_h_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_high& high) {
      return normal_trunc_h_log (y, mu, sigma, high, boost::math::policies::policy<>());
    }

    /**
     * The log of a density proportional to a truncated normal density for the given 
     * y, mean, standard deviation, and upper bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_high, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_high>::type
    normal_trunc_h_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_high& high, const Policy& /* pol */) {
      return normal_trunc_h_log (y, mu, sigma, high, Policy());
    }


    /**
     * The log of a density proportional to a truncated normal density for the given 
     * y, mean, standard deviation, and upper bound.
     * The standard deviation must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param mu The mean of the normal distribution.
     * @param sigma The standard deviation of the normal distribution. 
     * @param high Upper bound.
     * @throw std::domain_error if sigma is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_high Type of upper bound.
     */
    template <typename T_y, typename T_loc, typename T_scale, typename T_high>
    inline typename boost::math::tools::promote_args<T_y, T_loc, T_scale, T_high>::type
    normal_trunc_h_propto_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_high& high) {
      return normal_trunc_h_propto_log (y, mu, sigma, high, boost::math::policies::policy<>());
    }

    
  }
}
#endif
