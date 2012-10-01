#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__LOGISTIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__LOGISTIC_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    // Logistic(y|mu,sigma)    [sigma > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename return_type<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                 const Policy&) {
      static const char* function = "stan::prob::logistic_log(%1%)";
      
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return 0.0;


      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)      
      if (!check_finite(function, y, "Random variable", &logp, Policy()))
        return logp;
      if (!check_finite(function, mu, "Location parameter",
                        &logp, Policy()))
        return logp;
      if (!check_finite(function, sigma, "Scale parameter", &logp, 
                        Policy()))
        return logp;
      if (!check_positive(function, sigma, "Scale parameter",
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


      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      using stan::math::log1p;
      for (size_t n = 0; n < N; n++) {
	const double y_dbl = value_of(y_vec[n]);
	const double mu_dbl = value_of(mu_vec[n]);
	const double sigma_dbl = value_of(sigma_vec[n]);
	
	if (include_summand<propto,T_y,T_loc,T_scale>::value)
	  logp -= (y_dbl - mu_dbl)/sigma_dbl;
	if (include_summand<propto,T_scale>::value)
	  logp -= log(sigma_dbl);
	if (include_summand<propto,T_y,T_loc,T_scale>::value)
	  logp -= 2.0 * log1p(exp(-(y_dbl - mu_dbl)/sigma_dbl));

	if (!is_constant_struct<T_y>::value)
	  operands_and_partials.d_x1[n] += -1/sigma_dbl + 2 / (1 + exp(-(y_dbl - mu_dbl)/sigma_dbl)) * exp(-(y_dbl - mu_dbl)/sigma_dbl) / sigma_dbl;
	if (!is_constant_struct<T_loc>::value)
	  operands_and_partials.d_x2[n] += 1/sigma_dbl - 2 * exp(mu_dbl/sigma_dbl) / (sigma_dbl * (exp(mu_dbl/sigma_dbl) + exp(y_dbl/sigma_dbl)));
	if (!is_constant_struct<T_scale>::value)
	  operands_and_partials.d_x3[n] += (y_dbl - mu_dbl)/sigma_dbl/sigma_dbl - 1/sigma_dbl - 2*(y_dbl - mu_dbl) * exp(-(y_dbl - mu_dbl)/sigma_dbl) / sigma_dbl / sigma_dbl / (exp(-(y_dbl - mu_dbl)/sigma_dbl) + 1);
      }
      return operands_and_partials.to_var(logp);
    }

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return logistic_log<propto>(y,mu,sigma,stan::math::default_policy());
    }

    template <typename T_y, typename T_loc, typename T_scale,
              class Policy>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                 const Policy&) {
      return logistic_log<false>(y,mu,sigma,Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return logistic_log<false>(y,mu,sigma,stan::math::default_policy());
    }

  }
}
#endif
