#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__BETA_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__BETA_HPP__

#include <stan/prob/traits.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Beta(y|alpha,beta)  [alpha > 0;  beta > 0;  0 <= y <= 1]
    /**
     * The log of a beta density for y with the specified
     * prior sample sizes.
     * Prior sample sizes, alpha and beta, must be greater than 0.
     * y must be between 0 and 1 inclusive.
     * 
     * @param y A scalar variable.
     * @param alpha Prior sample size.
     * @param beta Prior sample size.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not between 0 and 1.
     * @tparam T_y Type of scalar outcome.
     * @tparam T_scale_succ Type of prior scale for successes.
     * @tparam T_scale_fail Type of prior scale for failures.
     * @error_policy
     *    @li alpha must be positive and finite.
     *    @li beta must be positive and finite.
     */
    template <bool propto,
              typename T_y, typename T_scale_succ, typename T_scale_fail,
              class Policy>
    typename
    boost::math::tools::promote_args<T_y,T_scale_succ,T_scale_fail>::type
    beta_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta, 
             const Policy&) {
      static const char* function = "stan::prob::beta_log<%1%>(%1%)";
      
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_scale_succ,T_scale_fail>::type lp;
      if (!check_finite(function, alpha,
                        "Prior success sample size plus 1, alpha,",
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, 
                          "Prior success sample size plus 1, alpha,",
                          &lp, Policy()))
        return lp;
      if (!check_finite(function, beta, 
                          "Prior failure sample size plus 1, beta,",
                          &lp, Policy()))
        return lp;
      if (!check_positive(function, beta, 
                          "Prior failure sample size plus 1, beta,",
                          &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "Random variate, y,", &lp, Policy()))
        return lp;
      
      if (y < 0.0 || y > 1.0)
        return LOG_ZERO;
      
      using stan::math::multiply_log;
      using stan::math::log1m;

      lp = 0.0;
      if (include_summand<propto,T_scale_succ,T_scale_fail>::value)
        lp += lgamma(alpha + beta);
      if (include_summand<propto,T_scale_succ>::value)
        lp -= lgamma(alpha);
      if (include_summand<propto,T_scale_fail>::value)
        lp -= lgamma(beta);
      if (include_summand<propto,T_y,T_scale_succ>::value)
        lp += multiply_log(alpha-1.0, y);
      if (include_summand<propto,T_y,T_scale_fail>::value)
        lp += (beta - 1.0) * log1m(y);
      return lp;
    }

    template <bool propto,
              typename T_y, typename T_scale_succ, typename T_scale_fail>
    inline 
    typename 
    boost::math::tools::promote_args<T_y,T_scale_succ,T_scale_fail>::type
    beta_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta) {
      return beta_log<propto>(y,alpha,beta,stan::math::default_policy());
    }

    template <typename T_y, typename T_scale_succ, typename T_scale_fail,
              class Policy>
    typename 
    boost::math::tools::promote_args<T_y,T_scale_succ,T_scale_fail>::type
    beta_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta, 
             const Policy&) {
      return beta_log<false>(y,alpha,beta,Policy());
    }

    template <typename T_y, typename T_scale_succ, typename T_scale_fail>
    inline 
    typename 
    boost::math::tools::promote_args<T_y,T_scale_succ,T_scale_fail>::type
    beta_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta) {
      return beta_log<false>(y,alpha,beta,stan::math::default_policy());
    }


    template <typename T_y, typename T_scale_succ, typename T_scale_fail,
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_scale_succ,T_scale_fail>::type
    beta_p(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta, 
           const Policy&) {
      static const char* function = "stan::prob::beta_p(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_scale_succ,T_scale_fail>::type lp;
      if (!check_finite(function, alpha,
                        "Prior success sample size plus 1, alpha,",
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, 
                          "Prior success sample size plus 1, alpha,",
                          &lp, Policy()))
        return lp;
      if (!check_finite(function, beta, 
                          "Prior failure sample size plus 1, beta,",
                          &lp, Policy()))
        return lp;
      if (!check_positive(function, beta, 
                          "Prior failure sample size plus 1, beta,",
                          &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "Random variate, y,", &lp, Policy()))
        return lp;
      
      if (y < 0.0)
        return 0;
      if (y > 1.0)
        return 1.0;
      
      return stan::math::ibeta(alpha, beta, y);
    }

    template <typename T_y, typename T_scale_succ, typename T_scale_fail>
    typename boost::math::tools::promote_args<T_y,T_scale_succ,T_scale_fail>::type
    beta_p(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta) {
      return beta_p(y,alpha,beta,stan::math::default_policy());
    }
  }
}
#endif
