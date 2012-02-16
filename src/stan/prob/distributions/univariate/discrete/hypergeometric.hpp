#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__HYPERGEOMETRIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__HYPERGEOMETRIC_HPP__

#include <stan/math/error_handling.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Hypergeometric(n|N,a,b)  [0 <= n <= a;  0 <= N-n <= b;  0 <= N <= a+b]
    // n: #white balls drawn;  N: #balls drawn;  
    // a: #white balls;  b: #black balls
    template <bool propto = false, 
              class Policy = stan::math::default_policy>
    inline double
    hypergeometric_log(const unsigned int n, 
                       const unsigned int N, 
                       const unsigned int a, 
                       const unsigned int b, 
                       const Policy& = Policy()) {
      static const char* function = "stan::prob::hypergeometric_log<%1%>(%1%)";

      using stan::math::check_finite;      
      using stan::math::check_bounded;
      using stan::math::check_greater;

      double lp(0.0);
      if (!check_bounded(function, n, 0U, a, "Number, n,", &lp, Policy()))
        return lp;
      if (!check_greater(function, N, n, "Number, N,", &lp, Policy()))
        return lp;
      if (!check_bounded(function, N-n, 0U, b, "Number, N-n,", &lp, Policy()))
        return lp;
      if (!check_bounded(function, N, 0U, a+b, "Number, N,", &lp, Policy()))
        return lp;
      
      
      if (include_summand<propto>::value)
        lp += math::binomial_coefficient_log(a,n)
          + math::binomial_coefficient_log(b,N-n)
          - math::binomial_coefficient_log(a+b,N);
      return lp;
    }

  }
}
#endif
