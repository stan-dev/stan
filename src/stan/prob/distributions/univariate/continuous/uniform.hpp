#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__UNIFORM_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__UNIFORM_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
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
    typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta, 
                const Policy&) {
      static const char* function = "stan::prob::uniform_log<%1%>(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_greater;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_low,T_high>::type lp(0.0);
      if(!check_not_nan(function, y, "y", &lp, Policy()))
        return lp;
      if (!check_finite(function, alpha, "Lower bound, alpha,", &lp, Policy()))
        return lp;
      if (!check_finite(function, beta, "Upper bound, beta,", &lp, Policy()))
        return lp;
      if (!check_greater(function, beta, alpha, "Upper bound, beta,",
                         &lp, Policy()))
        return lp;
      
      if (y < alpha || y > beta)
        return LOG_ZERO;
      
      if (include_summand<propto,T_low,T_high>::value)
        lp -= log(beta - alpha);
      return lp;
    }


    template <bool propto,
              typename T_y, typename T_low, typename T_high>
    inline
    typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      return uniform_log<propto>(y,alpha,beta,stan::math::default_policy());
    }

    template <typename T_y, typename T_low, typename T_high, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta, 
                const Policy&) {
      return uniform_log<false>(y,alpha,beta,Policy());
    }


    template <typename T_y, typename T_low, typename T_high>
    inline
    typename boost::math::tools::promote_args<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      return uniform_log<false>(y,alpha,beta,stan::math::default_policy());
    }

     
  }
}
#endif
