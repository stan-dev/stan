#ifndef __STAN__PROB__DISTRIBUTIONS__BETA_HPP__
#define __STAN__PROB__DISTRIBUTIONS__BETA_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>

namespace stan {
  namespace prob {

    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

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
     * @tparam T_y Type of scalar.
     * @tparam T_alpha Type of prior sample size for alpha.
     * @tparam T_beta Type of prior sample size for beta.
     */
    template <bool propto = false,
              typename T_y, typename T_alpha, typename T_beta, 
              class Policy = policy<> >
      inline typename promote_args<T_y,T_alpha,T_beta>::type
      beta_log(const T_y& y, const T_alpha& alpha, const T_beta& beta, const Policy& = Policy()) {
      static const char* function = "stan::prob::beta_log<%1%>(%1%)";

      typename promote_args<T_y,T_alpha,T_beta>::type lp(0.0);
      if(!stan::prob::check_positive(function, alpha, "Prior sample size, alpha,", &lp, Policy()))
        return lp;
      if(!stan::prob::check_positive(function, beta, "Prior sample size, beta,", &lp, Policy()))
        return lp;
      if(!stan::prob::check_bounded_x(function, y, 0, 1, &lp, Policy()))
        return lp;

      using stan::maths::multiply_log;
      using stan::maths::log1m;
      if (include_summand<propto,T_alpha,T_beta>::value)
        lp += lgamma(alpha + beta);
      if (include_summand<propto,T_alpha>::value)
        lp -= lgamma(alpha);
      if (include_summand<propto,T_beta>::value)
        lp -= lgamma(beta);
      if (include_summand<propto,T_y,T_alpha>::value)
        lp += multiply_log(alpha-1.0, y);
      if (include_summand<propto,T_y,T_beta>::value)
        lp += (beta - 1.0) * log1m(y);
      return lp;
    }

    // Beta(y|mu,theta)  [0 < mu < 1; theta > 0;  0 <= y <= 1]
    /**
     * The log of a beta density for y with the specified
     * mean and sample size.
     * Sample size, theta, must be greater than 0.
     * Mean must be between zero and one exclusive.
     * y must be between 0 and 1 inclusive.
     * 
     * @param y A scalar variable.
     * @param mu Prior mean.
     * @param theta Prior sample size.
     * @throw std::domain_error if mu * theta is not greater than 0.
     * @throw std::domain_error if (1 - mu) * theta is not greater than 0.
     * @throw std::domain_error if y is not between 0 and 1.
     * @tparam T_y Type of scalar.
     * @tparam T_mu Type of prior mean for mu.
     * @tparam T_theta Type of prior sample size for theta.
     */
    template <bool propto = false,
              typename T_y, typename T_mu, typename T_theta, 
              class Policy = policy<> >
      inline typename promote_args<T_y,T_mu,T_theta>::type
      beta_ls_log(const T_y& y, const T_mu& mu, const T_theta& theta, const Policy& = Policy()) {
        T_mu alpha = mu * theta;
        T_mu beta = (1.0 - mu) * theta;
        return beta_log(y, alpha, beta, Policy());
      }

  }
}
#endif
