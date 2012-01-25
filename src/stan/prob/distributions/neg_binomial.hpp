#ifndef __STAN__PROB__DISTRIBUTIONS__NEG_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__NEG_BINOMIAL_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/maths/special_functions.hpp>

namespace stan {
  namespace prob {
    // NegBinomial(n|alpha,beta)  [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto = false,
              typename T_shape, typename T_inv_scale, 
              class Policy = boost::math::policies::policy<> >
    inline typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type
    neg_binomial_log(const int n, const T_shape& alpha, const T_inv_scale& beta, const Policy& = Policy()) {
      static const char* function = "stan::prob::neg_binomial_log<%1%>(%1%)";
            
      typename boost::math::tools::promote_args<T_shape, T_inv_scale>::type lp(0.0);
      if (!check_nonnegative(function, n, "n", &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, "Shape, alpha,", &lp, Policy()))
        return lp;
      if(!check_positive(function, beta, "Inverse scale, beta,", &lp, Policy()))
        return lp;

      using stan::maths::multiply_log;
      using stan::maths::binomial_coefficient_log;
      
      if (include_summand<propto,T_shape>::value)
        lp += binomial_coefficient_log<T_shape>(n + alpha - 1.0, n);
      if (include_summand<propto,T_shape,T_inv_scale>::value)
        lp += multiply_log(alpha, beta) - (alpha + n) * log1p(beta);
      return lp;
    }

  }
}
#endif
