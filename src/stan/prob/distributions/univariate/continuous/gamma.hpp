#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__GAMMA_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__GAMMA_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    /**
     * The log of a gamma density for y with the specified
     * shape and inverse scale parameters.
     * Shape and inverse scale parameters must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Gamma}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\beta^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} \exp^{- \beta y} \right) \\
     &=& \alpha \log(\beta) - \log(\Gamma(\alpha)) + (\alpha - 1) \log(y) - \beta y\\
     & & \mathrm{where} \; y > 0
     \f}
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <bool propto = false,
              typename T_y, typename T_shape, typename T_inv_scale, 
              class Policy = stan::math::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_inv_scale>::type
    gamma_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta, const Policy& = Policy()) {
      static const char* function = "stan::prob::gamma_log<%1%>(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_shape,T_inv_scale>::type lp;
      if (!check_not_nan(function, y, "Random variate, y,", &lp, Policy()))
        return lp;
      if (!check_finite(function, alpha, "Shape parameter, alpha,", &lp, Policy())) 
        return lp;
      if (!check_positive(function, alpha, "Shape parameter, alpha,", &lp, Policy())) 
        return lp;
      if (!check_finite(function, beta, "Inverse scale parameter, beta,", &lp, Policy())) 
        return lp;
      if (!check_positive(function, beta, "Inverse scale parameter, beta,", &lp, Policy())) 
        return lp;
      
      if (y < 0)
	return LOG_ZERO;

      using boost::math::lgamma;
      using stan::math::multiply_log;

      lp = 0.0;
      if (include_summand<propto,T_shape>::value)
        lp -= lgamma(alpha);
      if (include_summand<propto,T_shape,T_inv_scale>::value)
        lp += multiply_log(alpha,beta);
      if (include_summand<propto,T_y,T_shape>::value)
        lp += multiply_log(alpha-1.0,y);
      if (include_summand<propto,T_y,T_inv_scale>::value)
        lp -= beta * y;
      return lp;
    }

    /**
     * The cumulative density function for a gamma distribution for y with the specified
     * shape and inverse scale parameters.
     *
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */    
    template <typename T_y, typename T_shape, typename T_inv_scale, 
              class Policy = stan::math::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_inv_scale>::type
    gamma_p(const T_y& y, const T_shape& alpha, const T_inv_scale& beta, const Policy& = Policy()){
      static const char* function = "stan::prob::gamma_p<%1%>(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_shape,T_inv_scale>::type result;
      if (!check_finite(function, y, "Random variate, y,", &result, Policy()))
        return result;
      if (!check_nonnegative(function, y, "Random variate, y,", &result, Policy()))
        return result;
      if (!check_finite(function, alpha, "Shape parameter, alpha,", &result, Policy())) 
        return result;
      if (!check_positive(function, alpha, "Shape parameter, alpha,", &result, Policy())) 
        return result;
      if (!check_finite(function, beta, "Inverse scale parameter, beta,", &result, Policy())) 
        return result;
      if (!check_positive(function, beta, "Inverse scale parameter, beta,", &result, Policy())) 
        return result;
      
      using boost::math::gamma_p;
      return gamma_p(alpha, y*beta);
    }

  }
}

#endif
