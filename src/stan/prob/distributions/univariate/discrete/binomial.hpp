#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BINOMIAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    // Binomial(n|N,theta)  [N >= 0;  0 <= n <= N;  0 <= theta <= 1]
    template <bool propto = false,
              typename T_n, typename T_N, typename T_prob, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_prob,T_n,T_N>::type
    binomial_log(const T_n& n, const T_N& N, const T_prob& theta, const Policy& = Policy()) {
      static const char* function = "stan::prob::binomial_log<%1%>(%1%)";
      
      using stan::maths::check_finite;
      using stan::maths::check_bounded;
      using stan::maths::check_nonnegative;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_prob,T_n,T_N>::type lp(0.0);
      if (!check_bounded(function, n, 0, N,
                         "Successes, n,",
                         &lp, Policy()))
        return lp;
      if (!check_finite(function, N,
                        "Population size, N,",
                        &lp, Policy()))
        return lp;      
      if (!check_nonnegative(function, N,
                             "Population size, N,",
                             &lp, Policy()))
        return lp;
      if (!check_finite(function, theta,
                        "Probability, theta,",
                        &lp, Policy()))
        return lp;
      if (!check_bounded(function, theta, 0, 1,
                         "Probability, theta,",
                         &lp, Policy()))
        return lp;

      using stan::maths::multiply_log;
      using stan::maths::binomial_coefficient_log;
      using stan::maths::log1m;

      if (include_summand<propto,T_n,T_N>::value)
        lp += binomial_coefficient_log<T_N,T_n>(N,n);
      if (include_summand<propto,T_n,T_prob>::value)
        lp += multiply_log(n,theta);
      if (include_summand<propto,T_n,T_N,T_prob>::value)
        lp += (N-n) * log1m(theta);
      return lp;
    }
  }
}
#endif
