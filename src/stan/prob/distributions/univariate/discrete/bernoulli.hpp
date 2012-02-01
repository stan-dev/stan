#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BERNOULLI_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BERNOULLI_HPP__

#include <stan/prob/traits.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {
  namespace prob {
    // Bernoulli(n|theta)   [0 <= n <= 1;   0 <= theta <= 1]
    template <bool propto = false,
              typename T_prob, 
              class Policy = stan::maths::default_policy> 
    inline typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_log(const unsigned int n, const T_prob& theta, const Policy& = Policy()) {
      static const char* function = "stan::prob::bernoulli_log<%1%>(%1%)";

      using stan::maths::check_finite;
      using stan::maths::check_bounded;

      T_prob lp;
      if (!check_finite(function, n, "n", &lp, Policy()))
        return lp;
      if (!check_bounded(function, n, 0, 1, "n", &lp, Policy()))
        return lp;
      if (!check_finite(function, theta, "Probability, theta,", &lp, Policy()))
        return lp;
      if (!check_bounded(function, theta, 0.0, 1.0, "Probability, theta,", &lp, Policy()))
        return lp;

      using stan::maths::log1m;
      
      if (include_summand<propto,T_prob>::value) {
        if (n == 1)
          return log(theta);
        else if (n == 0)
          return log1m(theta);
      }
      return 0.0;
    }


  }
}
#endif
