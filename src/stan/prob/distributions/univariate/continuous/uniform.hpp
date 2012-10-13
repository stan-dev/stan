#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__UNIFORM_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__UNIFORM_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // CONTINUOUS, UNIVARIATE DENSITIES
    /**
     * The log of a uniform density for the given 
     * y, lower, and upper bound. 
     *
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{U}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta)) &=& \log \left( \frac{1}{\beta-\alpha} \right) \\
     &=& \log (1) - \log (\beta - \alpha) \\
     &=& -\log (\beta - \alpha) \\
     & & \mathrm{ where } \; y \in [\alpha, \beta], \log(0) \; \mathrm{otherwise}
     \f}
     * 
     * @param y A scalar variable.
     * @param alpha Lower bound.
     * @param beta Upper bound.
     * @throw std::invalid_argument if the lower bound is greater than 
     *    or equal to the lower bound
     * @tparam T_y Type of scalar.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <bool propto,
              typename T_y, typename T_low, typename T_high, 
              class Policy>
    typename return_type<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta, 
                const Policy&) {
      static const char* function = "stan::prob::uniform_log(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_greater;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      typename return_type<T_y,T_low,T_high>::type logp(0.0);
      if(!check_not_nan(function, y, "Random variable", &logp, Policy()))
        return logp;
      if (!check_finite(function, alpha, "Lower bound parameter", &logp, Policy()))
        return logp;
      if (!check_finite(function, beta, "Upper bound parameter", &logp, Policy()))
        return logp;
      if (!check_greater(function, beta, alpha, "Upper bound parameter",
                         &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,alpha,beta,
				   "Random variable","Lower bound parameter","Upper bound parameter",
                                   &logp, Policy())))
        return logp;

      
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_low,T_high>::value)
	return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_low> alpha_vec(alpha);
      VectorView<const T_high> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);

      for (size_t n = 0; n < N; n++) {
	const double y_dbl = value_of(y_vec[n]);
	if (y_dbl < value_of(alpha_vec[n]) || y_dbl > value_of(beta_vec[n]))
	  return LOG_ZERO;
      }
      
      for (size_t n = 0; n < N; n++) {
	if (include_summand<propto,T_low,T_high>::value)
	  logp -= log(beta_vec[n] - alpha_vec[n]);
      }
      return logp;
    }


    template <bool propto,
              typename T_y, typename T_low, typename T_high>
    inline
    typename return_type<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      return uniform_log<propto>(y,alpha,beta,stan::math::default_policy());
    }

    template <typename T_y, typename T_low, typename T_high, 
              class Policy>
    inline
    typename return_type<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta, 
                const Policy&) {
      return uniform_log<false>(y,alpha,beta,Policy());
    }


    template <typename T_y, typename T_low, typename T_high>
    inline
    typename return_type<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      return uniform_log<false>(y,alpha,beta,stan::math::default_policy());
    }

     
  }
}
#endif
