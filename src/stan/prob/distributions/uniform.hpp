#ifndef __STAN__PROB__DISTRIBUTIONS__UNIFORM_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIFORM_HPP__

#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/traits.hpp>


namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;
    
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
    template <bool propto = false, 
              typename T_y, typename T_low, typename T_high, 
              class Policy = policy<> >
    inline typename promote_args<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta, const Policy& = Policy()) {
      static const char* function = "stan::prob::uniform_log<%1%>(%1%)";
      
      typename promote_args<T_y,T_low,T_high>::type lp(0.0);
      if(!stan::prob::check_not_nan(function, y, "y", &lp, Policy()))
        return lp;
      if(!stan::prob::check_bounds(function, alpha, beta, &lp, Policy()))
        return lp;
      
      if (y < alpha || y > beta)
        return LOG_ZERO;
      
      if (include_summand<propto,T_low,T_high>::value)
        lp -= log(beta - alpha);
      return lp;
    }
     
  }
}
#endif
