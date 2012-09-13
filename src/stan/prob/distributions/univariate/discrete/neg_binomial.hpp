#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // NegBinomial(n|alpha,beta)  [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto,
              typename T_shape, typename T_inv_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_log(const int n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta, 
                     const Policy&) {

      static const char* function = "stan::prob::neg_binomial_log<%1%>(%1%)";

      using stan::math::check_finite;      
      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_shape, T_inv_scale>::type lp;
      if (!check_nonnegative(function, n, "Failures variable", &lp, Policy()))
        return lp;
      if (!check_finite(function, alpha, "Shape parameter", &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, "Shape parameter", &lp, Policy()))
        return lp;
      if (!check_finite(function, beta, "Inverse scale parameter",
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, beta, "Inverse scale parameter", 
                          &lp, Policy()))
        return lp;
      
      using stan::math::multiply_log;
      using stan::math::binomial_coefficient_log;
      
      lp = 0.0;

      // Special case where negative binomial reduces to Poisson
      if (alpha > 1e10) {
        if (include_summand<propto>::value)
          lp -= lgamma(n + 1.0);
        if (include_summand<propto,T_shape>::value ||
            include_summand<propto,T_inv_scale>::value) {
          typename promote_args<T_shape, T_inv_scale>::type lambda;
          lambda = alpha / beta;
          lp += multiply_log(n, lambda) - lambda;
          return lp;
        }
      }
      // More typical cases
      if (include_summand<propto,T_shape>::value)
	if (n != 0)
	  lp += binomial_coefficient_log<T_shape>(n + alpha - 1.0, n);
      if (include_summand<propto,T_shape,T_inv_scale>::value)
	lp += -n * log1p(beta) + alpha * log(beta / (1 + beta));
      return lp;
    }

    template <bool propto,
              typename T_shape, typename T_inv_scale>
    inline
    typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_log(const int n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta) {
      return neg_binomial_log<propto>(n,alpha,beta,
                                      stan::math::default_policy());
    }

    template <typename T_shape, typename T_inv_scale, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_log(const int n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta, 
                     const Policy&) {
      return neg_binomial_log<false>(n,alpha,beta,Policy());
    }

    template <typename T_shape, typename T_inv_scale>
    inline
    typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_log(const int n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta) {
      return neg_binomial_log<false>(n,alpha,beta,
                                      stan::math::default_policy());
    }


  }
}
#endif
