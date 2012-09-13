#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__BETA_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__BETA_HPP__

#include <stan/agrad.hpp>
#include <stan/prob/traits.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the beta density for the specified scalar(s) given the specified
     * sample size(s). y, alpha, or beta can each either be scalar or std::vector.
     * Any vector inputs must be the same length.
     *
     * <p> The result log probability is defined to be the sum of
     * the log probabilities for each observation/alpha/beta triple.
     *
     * Prior sample sizes, alpha and beta, must be greater than 0.
     * 
     * @param y (Sequence of) scalar(s).
     * @param alpha (Sequence of) prior sample size(s).
     * @param beta (Sequence of) prior sample size(s).
     * @return The log of the product of densities.
     * @tparam T_y Type of scalar outcome.
     * @tparam T_scale_succ Type of prior scale for successes.
     * @tparam T_scale_fail Type of prior scale for failures.
     * @error_policy
     *    @li alpha must be positive and finite.
     *    @li beta must be positive and finite.
     */
    template <bool Prop,
              typename T_y, typename T_scale_succ, typename T_scale_fail,
              class Policy>
    typename return_type<T_y,T_scale_succ,T_scale_fail>::type
    beta_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta, 
             const Policy&) {
      static const char* function = "stan::prob::beta_log<%1%>(%1%)";
      
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::multiply_log;
      using stan::math::log1m;
      using stan::math::value_of;
      using stan::prob::include_summand;

      // check if no variables are involved and prop-to
      if (!include_summand<Prop,T_y,T_scale_succ,T_scale_fail>::value)
	return 0.0;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator

      double logp(0.0);
      
      // validate args (here done over var, which should be OK)
      if (!check_finite(function, alpha,
                        "First shape parameter",
                        &logp, Policy()))
        return logp;
      if (!check_positive(function, alpha, 
                          "First shape parameter",
                          &logp, Policy()))
        return logp;
      if (!check_finite(function, beta, 
                          "Second shape parameter",
                          &logp, Policy()))
        return logp;
      if (!check_positive(function, beta, 
                          "Second shape parameter",
                          &logp, Policy()))
        return logp;
      if (!check_not_nan(function, y, "Random variable", &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,alpha,beta,
				   "Random variable","First shape parameter","Second shape parameter",
                                   &logp, Policy())))
        return logp;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale_succ> alpha_vec(alpha);
      VectorView<const T_scale_fail> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);
      agrad::OperandsAndPartials<T_y, T_scale_succ, T_scale_fail>
        operands_and_partials(y, alpha, beta, y_vec, alpha_vec, beta_vec);

      for (size_t n = 0; n < N; n++) {
	// pull out values of arguments
	const double y_dbl = value_of(y_vec[n]);
	const double alpha_dbl = value_of(alpha_vec[n]);
	const double beta_dbl = value_of(beta_vec[n]);

	// reusable subexpressions values
	if (y_dbl < 0 || y_dbl > 1)
	  return 0.0;

	const double log_y = log(y_dbl);
	const double log1m_y = log1m(y_dbl);
	// log probability
	if (include_summand<Prop,T_scale_succ,T_scale_fail>::value)
	  logp += lgamma(alpha_dbl + beta_dbl);
	if (include_summand<Prop,T_scale_succ>::value)
	  logp -= lgamma(alpha_dbl);
	if (include_summand<Prop,T_scale_fail>::value)
	  logp -= lgamma(beta_dbl);
	if (include_summand<Prop,T_y,T_scale_succ>::value)
	  logp += (alpha_dbl-1.0) * log_y;
	if (include_summand<Prop,T_y,T_scale_fail>::value)
	  logp += (beta_dbl-1.0) * log1m_y;

	// gradients
	const double digamma_alpha_beta = boost::math::digamma(alpha_dbl + beta_dbl);
	const double digamma_alpha = boost::math::digamma(alpha_dbl);
	const double digamma_beta = boost::math::digamma(beta_dbl);

	if (!is_constant<typename is_vector<T_y>::type>::value)
	  operands_and_partials.d_x1[n] += (alpha_dbl-1)/y_dbl + (beta_dbl-1)/(y_dbl-1);
	if (!is_constant<typename is_vector<T_scale_succ>::type>::value)
	  operands_and_partials.d_x2[n] += log_y + digamma_alpha_beta - digamma_alpha;
	if (!is_constant<typename is_vector<T_scale_fail>::type>::value)
	  operands_and_partials.d_x3[n] += log1m_y + digamma_alpha_beta - digamma_beta;
      }
      return operands_and_partials.to_var(logp);
    }

    template <bool Prop,
              typename T_y, typename T_scale_succ, typename T_scale_fail>
    inline 
    typename return_type<T_y,T_scale_succ,T_scale_fail>::type
    beta_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta) {
      return beta_log<Prop>(y,alpha,beta,stan::math::default_policy());
    }

    template <typename T_y, typename T_scale_succ, typename T_scale_fail,
              class Policy>
    typename return_type<T_y,T_scale_succ,T_scale_fail>::type
    beta_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta, 
             const Policy&) {
      return beta_log<false>(y,alpha,beta,Policy());
    }

    template <typename T_y, typename T_scale_succ, typename T_scale_fail>
    inline 
    typename return_type<T_y,T_scale_succ,T_scale_fail>::type
    beta_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta) {
      return beta_log<false>(y,alpha,beta,stan::math::default_policy());
    }

    
    /**
     * Calculates the beta cumulative distribution function for the given
     * variate and scale variables.
     * 
     * @param y A scalar variate.
     * @param alpha Prior sample size.
     * @param beta Prior sample size.
     * @return The beta cdf evaluated at the specified arguments.
     * @tparam T_y Type of y.
     * @tparam T_scale_succ Type of alpha.
     * @tparam T_scale_fail Type of beta.
     * @tparam Policy Error-handling policy.
     */
    template <typename T_y, typename T_scale_succ, typename T_scale_fail,
              class Policy>
    typename return_type<T_y,T_scale_succ,T_scale_fail>::type
    beta_cdf(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta, 
	     const Policy&) {
      static const char* function = "stan::prob::beta_cdf(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_scale_succ,T_scale_fail>::type lp;
      if (!check_finite(function, alpha,
                        "First shape parameter",
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, 
                          "First shape parameter",
                          &lp, Policy()))
        return lp;
      if (!check_finite(function, beta, 
                          "Second shape parameter",
                          &lp, Policy()))
        return lp;
      if (!check_positive(function, beta, 
                          "Second shape parameter",
                          &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "Random variable", &lp, Policy()))
        return lp;
      
      if (y < 0.0)
        return 0;
      if (y > 1.0)
        return 1.0;
      
      return stan::math::ibeta(alpha, beta, y);
    }

    template <typename T_y, typename T_scale_succ, typename T_scale_fail>
    typename return_type<T_y,T_scale_succ,T_scale_fail>::type
    beta_cdf(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta) {
      return beta_cdf(y,alpha,beta,stan::math::default_policy());
    }
  }
}
#endif
