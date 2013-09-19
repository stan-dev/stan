#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__TRUNC_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__TRUNC_NORMAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include<boost/math/distributions.hpp>

namespace stan {
  
  namespace prob {
    /**
     * The log of the truncated normal density for the given y, 
     * mean, and standard deviation.  The standard deviation must be
     * greater than 0.
     *
     * \f{eqnarray*}{
     y &\sim& \mbox{\sf{N}} (\mu, \sigma^2) \\
     \log (p (y \,|\, \mu, \sigma) ) &=& \log \left( \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( - \frac{1}{2 \sigma^2} (y - \mu)^2 \right) \right) \\
     &=& \log (1) - \frac{1}{2}\log (2 \pi) - \log (\sigma) - \frac{(y - \mu)^2}{2 \sigma^2} - log(\Phi(\frac{\beta - \mu}{\sigma}) - \Phi(\frac{\alpha - \mu}{\sigma}))
     \f}
     * 
     * Errors are configured by policy.  All variables except alpha and beta 
     * must be finite, the scale must be strictly greater than zero and alpha < beta
     * 
     * @param y A scalar variate.
     * @param mu The location of the normal distribution.
     * @param sigma The scale of the normal distribution. 
     * @param alpha The lowerbound of the normal distribution. 
     * @param beta The upperbound of the normal distribution. 
     * @return The log of the normal density of the specified arguments.
     * @tparam propto Set to <code>true</code> if only calculated up
     * to a proportion.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @tparam T_alpha Type of lowerbound.
     * @tparam T_beta Type of upperbound.
     * @tparam Policy Error-handling policy.
     */
    template <bool propto, typename T_y, 
              typename T_loc, typename T_scale, 
              typename T_alpha, typename T_beta>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,
                                              T_alpha,T_beta>::type
    trunc_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                     const T_alpha& alpha, const T_beta& beta) {
      static const char* function = "stan::prob::trunc_normal_log(%1%)";
      
      using stan::math::check_greater;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      using boost::math::isinf;
      using boost::math::isfinite;
      using stan::math::Phi;
      
      typename promote_args<T_y,T_loc,T_scale,T_alpha,T_beta>::type lp(0.0);

      if (!check_not_nan(function, alpha, "Lower bound", 
                         &lp))
        return lp;
      if (!check_not_nan(function, beta, "Upper bound", 
                         &lp))
        return lp;
      if (!check_greater(function, beta, alpha, "Upper bound", 
                         &lp))
        return lp;
      
      if (y < alpha || y > beta) {
        lp = LOG_ZERO;
      }
      else {
        lp = normal_log<propto>(y,mu,sigma);
        if (include_summand<propto,T_loc,T_scale,T_alpha,T_beta>::value) {
          if (isinf(sigma)) 
            lp -= log(beta - alpha);
          else
            if (!isinf(beta) && !isinf(alpha)) 
              lp -= log(Phi((beta - mu)/sigma) - Phi((alpha - mu)/sigma));
            else if (isfinite(alpha)) 
              lp -= log(1.0 - Phi((alpha - mu)/sigma));
            else if (isfinite(beta)) 
              lp -= log(Phi((beta - mu)/sigma));
        }
      }
      
      return lp;
    }
    
    template <typename T_y, typename T_loc, typename T_scale, 
              typename T_alpha, typename T_beta>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,
                                              T_alpha,T_beta>::type
    trunc_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                     const T_alpha& alpha, const T_beta& beta) {
      return trunc_normal_log<false>(y,mu,sigma,alpha,beta);
    }
      
    template <class RNG>
    inline double
    trunc_normal_rng(const double mu,
                     const double sigma,
                     const double alpha,
                     const double beta,
                     RNG& rng) {
      using boost::variate_generator;

      static const char* function = "stan::prob::trunc_normal_rng(%1%)";
      
      using stan::math::check_greater;
      using stan::math::check_not_nan;

      if (!check_not_nan(function, alpha, "Lower bound")) 
        return 0;
      if (!check_not_nan(function, beta, "Upper bound"))
        return 0;
      if (!check_greater(function, beta, alpha, "Upper bound")) 
        return 0;

      double a = stan::prob::normal_rng(mu, sigma, rng);
      while(a > beta || a < alpha)
        a = stan::prob::normal_rng(mu,sigma,rng);
      return a;
    }
  }
}
#endif
