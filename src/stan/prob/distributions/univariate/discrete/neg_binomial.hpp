#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // NegBinomial(n|alpha,beta)  [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto,
	      typename T_n,
              typename T_shape, typename T_inv_scale, 
              class Policy>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_log(const T_n& n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta, 
                     const Policy&) {

      static const char* function = "stan::prob::neg_binomial_log(%1%)";

      using stan::math::check_finite;      
      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
	    && stan::length(alpha)
	    && stan::length(beta)))
	return 0.0;
      
      typename return_type<T_shape, T_inv_scale>::type logp(0.0);
      if (!check_nonnegative(function, n, "Failures variable", &logp, Policy()))
        return logp;
      if (!check_finite(function, alpha, "Shape parameter", &logp, Policy()))
        return logp;
      if (!check_positive(function, alpha, "Shape parameter", &logp, Policy()))
        return logp;
      if (!check_finite(function, beta, "Inverse scale parameter",
                        &logp, Policy()))
        return logp;
      if (!check_positive(function, beta, "Inverse scale parameter", 
                          &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,alpha,beta,
				   "Failures variable","Shape parameter","Inverse scale parameter",
                                   &logp, Policy())))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_shape,T_inv_scale>::value)
	return 0.0;

      using stan::math::multiply_log;
      using stan::math::binomial_coefficient_log;
      
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);

      for (size_t i = 0; i < size; i++) {
	// Special case where negative binomial reduces to Poisson
	if (alpha_vec[i] > 1e10) {
	  if (include_summand<propto>::value)
	    logp -= lgamma(n_vec[i] + 1.0);
	  if (include_summand<propto,T_shape>::value ||
	      include_summand<propto,T_inv_scale>::value) {
	    typename return_type<T_shape, T_inv_scale>::type lambda;
	    lambda = alpha_vec[i] / beta_vec[i];
	    logp += multiply_log(n_vec[i], lambda) - lambda;
	  }
	} else {
	// More typical cases
	  if (include_summand<propto,T_shape>::value)
	    if (n_vec[i] != 0)
	      logp += binomial_coefficient_log<typename scalar_type<T_shape>::type>
		(n_vec[i] + alpha_vec[i] - 1.0, n_vec[i]);
	  if (include_summand<propto,T_shape,T_inv_scale>::value)
	    logp += -n_vec[i] * log1p(beta_vec[i]) 
	      + alpha_vec[i] * log(beta_vec[i] / (1 + beta_vec[i]));
	}
      }
      return logp;
    }

    template <bool propto,
	      typename T_n,
              typename T_shape, typename T_inv_scale>
    inline
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_log(const T_n& n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta) {
      return neg_binomial_log<propto>(n,alpha,beta,
                                      stan::math::default_policy());
    }

    template <typename T_n,
	      typename T_shape, typename T_inv_scale, 
              class Policy>
    inline
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_log(const T_n& n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta, 
                     const Policy&) {
      return neg_binomial_log<false>(n,alpha,beta,Policy());
    }

    template <typename T_n, 
	      typename T_shape, typename T_inv_scale>
    inline
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_log(const T_n& n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta) {
      return neg_binomial_log<false>(n,alpha,beta,
                                      stan::math::default_policy());
    }


  }
}
#endif
