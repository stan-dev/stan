#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__HYPERGEOMETRIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__HYPERGEOMETRIC_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Hypergeometric(n|N,a,b)  [0 <= n <= a;  0 <= N-n <= b;  0 <= N <= a+b]
    // n: #white balls drawn;  N: #balls drawn;  
    // a: #white balls;  b: #black balls
    template <bool propto,
	      typename T_n,
	      typename T_N,
	      typename T_a,
	      typename T_b,
              class Policy>
    double
    hypergeometric_log(const T_n& n, 
                       const T_N& N, 
                       const T_a& a, 
                       const T_b& b, 
                       const Policy&) {
      static const char* function = "stan::prob::hypergeometric_log(%1%)";

      using stan::math::check_finite;      
      using stan::math::check_bounded;
      using stan::math::check_greater;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
	    && stan::length(N)
	    && stan::length(a)
	    && stan::length(b)))
	return 0.0;


      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_a> a_vec(a);
      VectorView<const T_b> b_vec(b);
      size_t size = max_size(n, N, a, b);
      
      double logp(0.0);
      if (!check_bounded(function, n, 0, a, "Successes variable", &logp, Policy()))
	return logp;
      if (!check_greater(function, N, n, "Draws parameter", &logp, Policy()))
	  return logp;
      for (size_t i = 0; i < size; i++) {
	if (!check_bounded(function, N_vec[i]-n_vec[i], 0, b_vec[i], "Draws parameter minus successes variable", &logp, Policy()))
	  return logp;
	if (!check_bounded(function, N_vec[i], 0, a_vec[i]+b_vec[i], "Draws parameter", &logp, Policy()))
	  return logp;
      }
      if (!(check_consistent_sizes(function,
				   n,N,a,b,
				   "Successes variable","Draws parameter","Successes in population parameter","Failures in population parameter",
				   &logp, Policy())))
	return logp;
      
      // check if no variables are involved and prop-to
      if (!include_summand<propto>::value)
	return 0.0;


      for (size_t i = 0; i < size; i++)
        logp += math::binomial_coefficient_log(a_vec[i],n_vec[i])
          + math::binomial_coefficient_log(b_vec[i],N_vec[i]-n_vec[i])
          - math::binomial_coefficient_log(a_vec[i]+b_vec[i],N_vec[i]);
      return logp;
    }


    template <bool propto,
	      typename T_n,
	      typename T_N,
	      typename T_a,
	      typename T_b>
    inline
    double
    hypergeometric_log(const T_n& n, 
                       const T_N& N, 
                       const T_a& a, 
                       const T_b& b) {
      return hypergeometric_log<propto>(n,N,a,b,stan::math::default_policy());
    }

    template <typename T_n,
	      typename T_N,
	      typename T_a,
	      typename T_b,
	      class Policy>
    inline
    double
    hypergeometric_log(const T_n& n, 
                       const T_N& N, 
                       const T_a& a, 
                       const T_b& b, 
                       const Policy&) {
      return hypergeometric_log<false>(n,N,a,b,Policy());
    }

    template <typename T_n,
	      typename T_N,
	      typename T_a,
	      typename T_b>
    inline
    double
    hypergeometric_log(const T_n& n, 
                       const T_N& N, 
                       const T_a& a, 
                       const T_b& b) {
      return hypergeometric_log<false>(n,N,a,b,stan::math::default_policy());
    }


  }
}
#endif
