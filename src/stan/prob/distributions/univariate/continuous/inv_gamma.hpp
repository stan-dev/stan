#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__INV_GAMMA_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__INV_GAMMA_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
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
    template <bool propto,
              typename T_y, typename T_shape, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta, 
                  const Policy&) {
      static const char* function = "stan::prob::inv_gamma_log(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_shape,T_scale>::type lp;
      if (!check_not_nan(function, y, "Random variable", &lp, Policy()))
        return lp;
      if (!check_finite(function, alpha, "Shape parameter", 
                        &lp, Policy())) 
        return lp;
      if (!check_positive(function, alpha, "Shape parameter",
                          &lp, Policy())) 
        return lp;
      if (!check_finite(function, beta, "Scale parameter",
                        &lp, Policy())) 
        return lp;
      if (!check_positive(function, beta, "Scale parameter", 
                          &lp, Policy())) 
        return lp;

      if (y <= 0)
        return LOG_ZERO;

      using boost::math::lgamma;
      using stan::math::multiply_log;
      
      lp = 0.0;
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

    template <bool propto,
              typename T_y, typename T_shape, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta) {
      return inv_gamma_log<propto>(y,alpha,beta,stan::math::default_policy());
    }

    template <typename T_y, typename T_shape, typename T_scale, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta, 
                  const Policy&) {
      return inv_gamma_log<false>(y,alpha,beta,Policy());
    }

    template <typename T_y, typename T_shape, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta) {
      return inv_gamma_log<false>(y,alpha,beta,stan::math::default_policy());
    }


          
  }
}

#endif
