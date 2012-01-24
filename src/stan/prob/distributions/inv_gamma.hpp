#ifndef __STAN__PROB__DISTRIBUTIONS__INV_GAMMA_HPP__
#define __STAN__PROB__DISTRIBUTIONS__INV_GAMMA_HPP__

#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    /**
     * The log of an inverse gamma density for y with the specified
     * shape and scale parameters.
     * Shape and scale parameters must be greater than 0.
     * y must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_scale Type of scale.
     */
    template <bool propto = false,
              typename T_y, typename T_shape, typename T_scale, 
              class Policy = boost::math::policies::policy<> >
      inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
      inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta, 
                    const Policy& = Policy()) {
      static const char* function = "stan::prob::inv_gamma_log<%1%>(%1%)";

      using boost::math::lgamma;
      using stan::maths::multiply_log;

      typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type lp(0.0);
      if (!stan::prob::check_positive(function, alpha, "Shape parameter, alpha,", &lp, Policy())) 
        return lp;
      if (!stan::prob::check_positive(function, beta, "Scale parameter, beta,", &lp, Policy())) 
        return lp;
      if (!stan::prob::check_positive(function, y, "Random variate, y,", &lp, Policy()))
        return lp;

      if (include_summand<propto,T_shape>::value)
        lp -= lgamma(alpha);
      if (include_summand<propto,T_shape,T_scale>::value)
        lp += multiply_log(alpha,beta);
      if (include_summand<propto,T_y,T_shape>::value)
        lp -= multiply_log(alpha+1.0, y);
      if (include_summand<propto,T_y,T_scale>::value)
        lp -= beta / y;
      return lp;
    }
          
  }
}

#endif
