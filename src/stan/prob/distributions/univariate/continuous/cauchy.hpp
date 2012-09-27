#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CAUCHY_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CAUCHY_HPP__

#include <stan/agrad.hpp>
#include <stan/prob/traits.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the Cauchy density for the specified scalar(s) given the specified
     * location parameter(s) and scale parameter(s). y, mu, or sigma can each either be scalar or std::vector.
     * Any vector inputs must be the same length.
     *
     * <p> The result log probability is defined to be the sum of
     * the log probabilities for each observation/mu/sigma triple.
     *
     * @param y (Sequence of) scalar(s).
     * @param mu (Sequence of) location(s).
     * @param sigma (Sequence of) scale(s).
     * @return The log of the product of densities.
     * @tparam T_y Type of scalar outcome.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @error_policy
     *    @li sigma must be positive.
     */
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
               const Policy&) {
      static const char* function = "stan::prob::cauchy_log(%1%)";
      
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return 0.0;
      
      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      if (!check_not_nan(function, y, "Random variable", &logp, Policy()))
        return logp;
      if (!check_finite(function, mu, "Location parameter", 
                        &logp, Policy()))
        return logp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &logp, Policy()))
        return logp;
      if (!check_finite(function, sigma, "Scale parameter", 
                        &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,
				   "Random variable","Location parameter","Scale parameter",
                                   &logp, Policy())))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;

      using stan::math::log1p;
      using stan::math::square;
      
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      DoubleVectorView<true, T_scale> inv_sigma(length(sigma));
      DoubleVectorView<true, T_scale> sigma_squared(length(sigma));
      DoubleVectorView<include_summand<propto,T_scale>::value,T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
	const double sigma_dbl = value_of(sigma_vec[i]);
        inv_sigma[i] = 1.0 / sigma_dbl;
	sigma_squared[i] = sigma_dbl * sigma_dbl;
	if (include_summand<propto,T_scale>::value) {
	  log_sigma[i] = log(sigma_dbl);
	}
      }

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> operands_and_partials(y, mu, sigma);

      for (size_t n = 0; n < N; n++) {
	// pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
	
	// reusable subexpression values
	const double y_minus_mu
	  = y_dbl - mu_dbl;
	const double y_minus_mu_squared
	  = y_minus_mu * y_minus_mu;
        const double y_minus_mu_over_sigma 
          = y_minus_mu * inv_sigma[n];
        const double y_minus_mu_over_sigma_squared 
          = y_minus_mu_over_sigma * y_minus_mu_over_sigma;

	// log probability
	if (include_summand<propto>::value)
	  logp += NEG_LOG_PI;
	if (include_summand<propto,T_scale>::value)
	  logp -= log_sigma[n];
	if (include_summand<propto,T_y,T_loc,T_scale>::value)
	  logp -= log1p(y_minus_mu_over_sigma_squared);
	
        // gradients
	if (!is_constant<typename is_vector<T_y>::type>::value)
	  operands_and_partials.d_x1[n] -= 2 * y_minus_mu / (sigma_squared[n] + y_minus_mu_squared);
        if (!is_constant<typename is_vector<T_loc>::type>::value)
          operands_and_partials.d_x2[n] += 2 * y_minus_mu / (sigma_squared[n] + y_minus_mu_squared);
        if (!is_constant<typename is_vector<T_scale>::type>::value)
          operands_and_partials.d_x3[n] += (y_minus_mu_squared - sigma_squared[n]) * inv_sigma[n] / (sigma_squared[n] + y_minus_mu_squared);
      }
      return operands_and_partials.to_var(logp);
    }


    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return cauchy_log<propto>(y,mu,sigma,stan::math::default_policy());
    }

    template <typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
               const Policy&) {
      return cauchy_log<false>(y,mu,sigma,Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return cauchy_log<false>(y,mu,sigma,stan::math::default_policy());
    }


    /** 
     * Calculates the cauchy cumulative distribution function for
     * the given variate, location, and scale.
     *
     * \f$\frac{1}{\pi}\arctan\left(\frac{y-\mu}{\sigma}\right) + \frac{1}{2}\f$ 
     *
     * Errors are configured by policy.  All variables must be finite
     * and the scale must be strictly greater than zero.
     *
     * @param y A scalar variate.
     * @param mu The location parameter.
     * @param sigma The scale parameter.
     * 
     * @return 
     */
    template <typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma, 
               const Policy&) {
      static const char* function = "stan::prob::cauchy_cdf(%1%)";
      
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if(!check_not_nan(function, y, "Random variable", &lp, Policy()))
        return lp;
      if(!check_finite(function, mu, "Location parameter", 
                       &lp, Policy()))
        return lp;
      if(!check_finite(function, sigma, "Scale parameter", 
                       &lp, Policy()))
        return lp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &lp, Policy()))
        return lp;

      using std::atan2;
      using stan::math::pi;

      return atan2(y-mu, sigma) / pi() + 0.5;
   }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return cauchy_cdf(y, mu, sigma, stan::math::default_policy());
    }
  }
}
#endif
