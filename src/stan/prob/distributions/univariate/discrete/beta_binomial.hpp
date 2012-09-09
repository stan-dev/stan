#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BETA_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BETA_BINOMIAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    // BetaBinomial(n|alpha,beta) [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto,
              typename T_size1, 
	      typename T_size2, 
              class Policy>
    typename boost::math::tools::promote_args<T_size1,T_size2>::type
    beta_binomial_log(const int n, 
                      const int N, 
                      const T_size1& alpha, 
                      const T_size2& beta, 
                      const Policy&) {
      static const char* function = "stan::prob::beta_binomial_log<%1%>(%1%)";

      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_size1,T_size2>::type lp;
      if (!check_nonnegative(function, N, "Population size, N,", &lp, Policy()))
        return lp;
      if (!check_finite(function, alpha, "Prior sample size, alpha,", &lp, 
                        Policy()))
        return lp;
      if (!check_positive(function, alpha, "Prior sample size, alpha,", 
                          &lp, Policy()))
        return lp;
      if (!check_finite(function, beta, "Prior sample size, beta,",
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, beta, "Prior sample size, beta,", 
                          &lp, Policy()))
        return lp;
      
      if (n < 0 || n > N)
        return LOG_ZERO;
      
      using stan::math::lbeta;
      using stan::math::binomial_coefficient_log;

      lp = 0.0;
      if (include_summand<propto>::value)
        lp += binomial_coefficient_log(N,n);
      if (include_summand<propto,T_size1,T_size2>::value)
        lp += lbeta(n + alpha, N - n + beta) - lbeta(alpha,beta);
      return lp;
    }

    template <bool propto,
              typename T_size1,
	      typename T_size2>
    typename boost::math::tools::promote_args<T_size1,T_size2>::type
    beta_binomial_log(const int n, const int N, 
                      const T_size1& alpha, const T_size2& beta) {
      return beta_binomial_log<propto>(n,N,alpha,beta,
                                       stan::math::default_policy());
    }

    template <typename T_size1,
	      typename T_size2, 
              class Policy>
    typename boost::math::tools::promote_args<T_size1,T_size2>::type
    inline
    beta_binomial_log(const int n, const int N, 
                      const T_size1& alpha, const T_size2& beta, 
                      const Policy&) {
      return beta_binomial_log<false>(n,N,alpha,beta,Policy());
    }

    template <typename T_size1,
	      typename T_size2>
    typename boost::math::tools::promote_args<T_size1,T_size2>::type
    beta_binomial_log(const int n, const int N, 
                      const T_size1& alpha, const T_size2& beta) {
      return beta_binomial_log<false>(n,N,alpha,beta,
                                      stan::math::default_policy());
    }

  }
}
#endif
