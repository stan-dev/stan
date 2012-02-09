#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__POISSON_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__POISSON_HPP__

#include <stan/prob/constants.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    // Poisson(n|lambda)  [lambda > 0;  n >= 0]
    template <bool propto = false,
              typename T_rate, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_rate>::type
    poisson_log(const unsigned int n, const T_rate& lambda, 
                const Policy& = Policy()) {

      static const char* function = "stan::prob::poisson_log<%1%>(%1%)";
      
      using stan::maths::check_not_nan;
      using stan::maths::check_nonnegative;
      using boost::math::tools::promote_args;

      typename promote_args<T_rate>::type lp;
      if (!check_nonnegative(function, n, "Number n", &lp, Policy()))
        return lp;
      if (!check_not_nan(function, lambda,
                         "Rate parameter, lambda,", &lp, Policy()))
        return lp;
      if (!check_nonnegative(function, lambda,
                             "Rate parameter, lambda,", &lp, Policy()))
        return lp;
      
      if (lambda == 0)
        return LOG_ZERO;
      
      using stan::maths::multiply_log;
      if (std::isinf(lambda))
        return LOG_ZERO;

      lp = 0.0;
      if (include_summand<propto>::value)
        lp -= lgamma(n + 1.0);
      if (include_summand<propto,T_rate>::value)
        lp += multiply_log(n, lambda) - lambda;
      return lp;
    }
    


  }
}
#endif
