#ifndef __STAN__PROB__DISTRIBUTIONS__HYPERGEOMETRIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__HYPERGEOMETRIC_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {
  namespace prob {
    // Hypergeometric(n|N,a,b)  [0 <= n <= a;  0 <= N-n <= b;  0 <= N <= a+b]
    // n: #white balls drawn;  N: #balls drawn;  a: #white balls;  b: #black balls
    template <bool propto = false, 
              class Policy = boost::math::policies::policy<> >
    inline double
    hypergeometric_log(const unsigned int n, const unsigned int N, 
                       const unsigned int a, const unsigned int b, const Policy& = Policy()) {
      static const char* function = "stan::prob::hypergeometric_log<%1%>(%1%)";
      
      double lp(0.0);
      if (!check_bounded(function, n, 0U, a, "Number, n,", &lp, Policy()))
        return lp;
      if (!check_greater(function, N, n, "Number, N,", &lp, Policy()))
        return lp;
      if (!check_bounded(function, N-n, 0U, b, "Number, N-n,", &lp, Policy()))
        return lp;
      if(!check_bounded(function, N, 0U, a+b, "Number, N,", &lp, Policy()))
        return lp;
      
      
      if (include_summand<propto>::value)
        lp += maths::binomial_coefficient_log(a,n)
          + maths::binomial_coefficient_log(b,N-n)
          - maths::binomial_coefficient_log(a+b,N);
      return lp;
    }

  }
}
#endif
