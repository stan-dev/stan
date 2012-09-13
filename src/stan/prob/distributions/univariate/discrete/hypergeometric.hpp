#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__HYPERGEOMETRIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__HYPERGEOMETRIC_HPP__

#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Hypergeometric(n|N,a,b)  [0 <= n <= a;  0 <= N-n <= b;  0 <= N <= a+b]
    // n: #white balls drawn;  N: #balls drawn;  
    // a: #white balls;  b: #black balls
    template <bool propto,
              class Policy>
    double
    hypergeometric_log(const unsigned int n, 
                       const unsigned int N, 
                       const unsigned int a, 
                       const unsigned int b, 
                       const Policy&) {
      static const char* function = "stan::prob::hypergeometric_log(%1%)";

      using stan::math::check_finite;      
      using stan::math::check_bounded;
      using stan::math::check_greater;

      double lp(0.0);
      if (!check_bounded(function, n, 0U, a, "Successes variable", &lp, Policy()))
        return lp;
      if (!check_greater(function, N, n, "Population size parameter", &lp, Policy()))
        return lp;
      if (!check_bounded(function, N-n, 0U, b, "Population size parameter minus success variable,", &lp, Policy()))
        return lp;
      if (!check_bounded(function, N, 0U, a+b, "Population size parameter", &lp, Policy()))
        return lp;
      
      
      if (include_summand<propto>::value)
        lp += math::binomial_coefficient_log(a,n)
          + math::binomial_coefficient_log(b,N-n)
          - math::binomial_coefficient_log(a+b,N);
      return lp;
    }


    template <bool propto>
    inline
    double
    hypergeometric_log(const unsigned int n, 
                       const unsigned int N, 
                       const unsigned int a, 
                       const unsigned int b) {
      return hypergeometric_log<propto>(n,N,a,b,stan::math::default_policy());
    }

    template <class Policy>
    inline
    double
    hypergeometric_log(const unsigned int n, 
                       const unsigned int N, 
                       const unsigned int a, 
                       const unsigned int b, 
                       const Policy&) {
      return hypergeometric_log<false>(n,N,a,b,Policy());
    }

    inline
    double
    hypergeometric_log(const unsigned int n, 
                       const unsigned int N, 
                       const unsigned int a, 
                       const unsigned int b) {
      return hypergeometric_log<false>(n,N,a,b,stan::math::default_policy());
    }


  }
}
#endif
