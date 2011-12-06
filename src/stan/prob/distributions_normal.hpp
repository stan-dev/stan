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
    
  }
}


#endif
