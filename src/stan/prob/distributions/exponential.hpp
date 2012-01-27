#ifndef __STAN__PROB__DISTRIBUTIONS__EXPONENTIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__EXPONENTIAL_HPP__

#include <stan/prob/traits.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {
  namespace prob {
    /**
     * The log of an exponential density for y with the specified
     * inverse scale parameter.
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
       y 
       &\sim& 
       \mbox{\sf{Expon}}(\beta) \\
       \log (p (y \,|\, \beta) )
       &=& 
       \log \left( \beta \exp^{-\beta y} \right) \\
       &=& 
       \log (\beta) - \beta y \\
       & & 
       \mathrm{where} \; y > 0
     \f}
     *
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <bool propto = false,
              typename T_y, typename T_inv_scale, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_inv_scale>::type
    exponential_log(const T_y& y, const T_inv_scale& beta, 
                    const Policy& = Policy()) {
      static const char* function = "stan::prob::exponential_log<%1%>(%1%)";

      using stan::maths::check_positive;
      using stan::maths::check_not_nan;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_inv_scale>::type lp(0.0);
      if(!check_positive(function, beta, "Inverse scale", &lp, Policy()))
        return lp;
      if(!check_not_nan(function, y, "Random variate y", &lp, Policy()))
        return lp;
      
      if (include_summand<propto,T_inv_scale>::value)
        lp += log(beta);
      if (include_summand<propto,T_y,T_inv_scale>::value)
        lp -= beta * y;
      return lp;
    }
    
    /**
     * Calculates the exponential cumulative distribution function for
     * the given y and beta.
     *
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     * @tparam Policy Error-handling policy.
     */
    template <typename T_y, 
              typename T_inv_scale, 
              class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_y,T_inv_scale>::type
    exponential_p(const T_y& y, 
                  const T_inv_scale& beta, 
                  const Policy& = Policy()) {

      static const char* function = "stan::prob::exponential_p<%1%>(%1%)";

      using boost::math::tools::promote_args;
      using stan::maths::check_positive;
      using stan::maths::check_not_nan;

      // FIXME(?possible?): throw away lp if succeeds tests and y >= 0

      typename promote_args<T_y,T_inv_scale>::type lp;
      if (!check_positive(function, beta, "Inverse scale", &lp, Policy()))
        return lp;

      if (!check_not_nan(function, y, "Random variate y", &lp, Policy()))
        return lp;
      
      if (y < 0)
        return 1.0;

      return 1.0 - exp(-beta * y);
    }


  }
}

#endif
