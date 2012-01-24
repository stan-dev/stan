#ifndef __STAN__PROB__DISTRIBUTIONS__BETA_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__BETA_BINOMIAL_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>
#include <stan/maths/special_functions.hpp>

namespace stan {
  namespace prob {

    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // BetaBinomial(n|alpha,beta) [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto = false,
              typename T_size, 
              class Policy = policy<> >
    inline typename promote_args<T_size>::type
    beta_binomial_log(const int n, const int N, const T_size& alpha, const T_size& beta,
                      const Policy& = Policy()) {
      static const char* function = "stan::prob::beta_binomial_log<%1%>(%1%)";

      typename promote_args<T_size>::type lp(0.0);
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

      if (include_summand<propto>::value)
        lp += binomial_coefficient_log(N,n);
      if (include_summand<propto,T_size>::value)
        lp += lbeta(n + alpha, N - n + beta) - lbeta(alpha,beta);
      return lp;
    }

  }
}
#endif
