#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BINOMIAL_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // Binomial(n|N,theta)  [N >= 0;  0 <= n <= N;  0 <= theta <= 1]
    template <bool propto,
              typename T_prob, 
              class Policy>
    typename boost::math::tools::promote_args<T_prob>::type
    binomial_log(const int n, 
                 const int N, 
                 const T_prob& theta, 
                 const Policy&) {

      static const char* function = "stan::prob::binomial_log<%1%>(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_bounded;
      using stan::math::check_nonnegative;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_prob>::type lp(0.0);
      if (!check_bounded(function, n, 0, N,
                         "Successes variable",
                         &lp, Policy()))
        return lp;
      if (!check_nonnegative(function, N,
                             "Population size parameter",
                             &lp, Policy()))
        return lp;
      if (!check_finite(function, theta,
                        "Probability parameter",
                        &lp, Policy()))
        return lp;
      if (!check_bounded(function, theta, 0.0, 1.0,
                         "Probability parameter",
                         &lp, Policy()))
        return lp;

      using stan::math::multiply_log;
      using stan::math::binomial_coefficient_log;
      using stan::math::log1m;

      if (include_summand<propto>::value)
        lp += binomial_coefficient_log(N,n);
      if (include_summand<propto,T_prob>::value) 
        lp += multiply_log(n,theta)
          + (N - n) * log1m(theta);
      return lp;
    }

    template <bool propto,
              typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    binomial_log(const int n, 
                 const int N, 
                 const T_prob& theta) {
      return binomial_log<propto>(n,N,theta,stan::math::default_policy());
    }


    template <typename T_prob, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    binomial_log(const int n, 
                 const int N, 
                 const T_prob& theta, 
                 const Policy&) {
      return binomial_log<false>(n,N,theta,Policy());
    }


    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    binomial_log(const int n, 
                 const int N, 
                 const T_prob& theta) {
      return binomial_log<false>(n,N,theta,stan::math::default_policy());
    }


    
  }
}
#endif
