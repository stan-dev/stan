#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BETA_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BETA_BINOMIAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    // BetaBinomial(n|alpha,beta) [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto = false,
              typename T_size, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_size>::type
    beta_binomial_log(const int n, const int N, const T_size& alpha, const T_size& beta, 
                      const Policy& = Policy()) {
      static const char* function = "stan::prob::beta_binomial_log<%1%>(%1%)";

      using stan::maths::check_nonnegative;
      using stan::maths::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_size>::type lp;
      if (!check_nonnegative(function, N, "Sample size, N,", &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, "Prior sample size, alpha,", &lp, Policy()))
        return lp;
      if (!check_positive(function, beta, "Prior sample size, beta,", &lp, Policy()))
        return lp;
      
      if (n < 0 || n > N)
        return LOG_ZERO;
      
      using stan::maths::lbeta;
      using stan::maths::binomial_coefficient_log;

      lp = 0.0;
      if (include_summand<propto>::value)
        lp += binomial_coefficient_log(N,n);
      if (include_summand<propto,T_size>::value)
        lp += lbeta(n + alpha, N - n + beta) - lbeta(alpha,beta);
      return lp;
    }

  }
}
#endif
