#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BERNOULLI_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BERNOULLI_HPP__

#include <stan/prob/traits.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Bernoulli(n|theta)   [0 <= n <= 1;   0 <= theta <= 1]
    template <bool propto,
              typename T_prob, 
              class Policy>
    typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_log(const int n, 
                  const T_prob& theta, 
                  const Policy&) {
      static const char* function = "stan::prob::bernoulli_log<%1%>(%1%)";

      using stan::math::check_finite;
      using stan::math::check_bounded;

      T_prob lp;
      if (!check_bounded(function, n, 0, 1, "n", &lp, Policy()))
        return lp;
      if (!check_finite(function, theta, "Probability, theta,", &lp, Policy()))
        return lp;
      if (!check_bounded(function, theta, 0.0, 1.0,
                         "Probability, theta,", &lp, Policy()))
        return lp;

      using stan::math::log1m;
      
      if (include_summand<propto,T_prob>::value) {
        if (n == 1)
          return log(theta);
        else if (n == 0)
          return log1m(theta);
      }
      return 0.0;
    }


    template <bool propto,
              typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_log(const int n, 
                  const T_prob& theta) {
      return bernoulli_log<propto>(n,theta,stan::math::default_policy());
    }


    template <typename T_prob, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_log(const int n, 
                  const T_prob& theta, 
                  const Policy&) {
      return bernoulli_log<false>(n,theta,Policy());
    }


    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_log(const int n, 
                  const T_prob& theta) {
      return bernoulli_log<false>(n,theta,stan::math::default_policy());
    }


  }
}
#endif
