#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__POISSON_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__POISSON_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // Poisson(n|lambda)  [lambda > 0;  n >= 0]
    template <bool propto,
              typename T_rate, 
              class Policy>
    typename boost::math::tools::promote_args<T_rate>::type
    poisson_log(const unsigned int n, const T_rate& lambda, 
                const Policy&) {

      static const char* function = "stan::prob::poisson_log<%1%>(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
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
        return n == 0 ? 0 : LOG_ZERO;

      using stan::math::multiply_log;
      if (std::isinf(lambda))
        return LOG_ZERO;

      lp = 0.0;
      if (include_summand<propto>::value)
        lp -= lgamma(n + 1.0);
      if (include_summand<propto,T_rate>::value)
        lp += multiply_log(n, lambda) - lambda;
      return lp;
    }
    
    template <bool propto,
              typename T_rate>
    inline
    typename boost::math::tools::promote_args<T_rate>::type
    poisson_log(const unsigned int n, const T_rate& lambda) {
      return poisson_log<propto>(n,lambda,stan::math::default_policy());
    }


    template <typename T_rate, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_rate>::type
    poisson_log(const unsigned int n, const T_rate& lambda, 
                const Policy&) {
      return poisson_log<false>(n,lambda,Policy());
    }


    template <typename T_rate>
    inline
    typename boost::math::tools::promote_args<T_rate>::type
    poisson_log(const unsigned int n, const T_rate& lambda) {
      return poisson_log<false>(n,lambda,stan::math::default_policy());
    }


  }
}
#endif
