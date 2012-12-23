#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BINOMIAL_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Binomial(n|N,theta)  [N >= 0;  0 <= n <= N;  0 <= theta <= 1]
    template <bool propto,
	      typename T_n,
	      typename T_N,
              typename T_prob, 
              class Policy>
    typename return_type<T_prob>::type
    binomial_log(const T_n& n, 
                 const T_N& N, 
                 const T_prob& theta, 
                 const Policy&) {

      static const char* function = "stan::prob::binomial_log(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_bounded;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
	    && stan::length(N)
	    && stan::length(theta)))
	return 0.0;

      typename return_type<T_prob>::type logp(0.0);
      if (!check_bounded(function, n, 0, N,
                         "Successes variable",
                         &logp, Policy()))
        return logp;
      if (!check_nonnegative(function, N,
                             "Population size parameter",
                             &logp, Policy()))
        return logp;
      if (!check_finite(function, theta,
                        "Probability parameter",
                        &logp, Policy()))
        return logp;
      if (!check_bounded(function, theta, 0.0, 1.0,
                         "Probability parameter",
                         &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,N,theta,
				   "Successes variable","Population size parameter","Probability parameter",
                                   &logp, Policy())))
        return logp;


      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_prob>::value)
	return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_prob> theta_vec(theta);
      size_t size = max_size(n, N, theta);
      //agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);
      
      using stan::math::multiply_log;
      using stan::math::binomial_coefficient_log;
      using stan::math::log1m;
	
      for (size_t i = 0; i < size; i++) {
	if (include_summand<propto>::value)
	  logp += binomial_coefficient_log(N_vec[i],n_vec[i]);
	if (include_summand<propto,T_prob>::value) 
	  logp += multiply_log(n_vec[i],theta_vec[i])
	    + (N_vec[i] - n_vec[i]) * log1m(theta_vec[i]);
      }
      return logp;
    }

    template <bool propto,
	      typename T_n,
	      typename T_N,
              typename T_prob>
    inline
    typename return_type<T_prob>::type
    binomial_log(const T_n& n, 
                 const T_N& N, 
                 const T_prob& theta) {
      return binomial_log<propto>(n,N,theta,stan::math::default_policy());
    }


    template <typename T_n,
	      typename T_N,
	      typename T_prob, 
              class Policy>
    inline
    typename return_type<T_prob>::type
    binomial_log(const T_n& n, 
                 const T_N& N, 
                 const T_prob& theta, 
                 const Policy&) {
      return binomial_log<false>(n,N,theta,Policy());
    }


    template <typename T_n, 
	      typename T_N,
	      typename T_prob>
    inline
    typename return_type<T_prob>::type
    binomial_log(const T_n& n, 
                 const T_N& N, 
                 const T_prob& theta) {
      return binomial_log<false>(n,N,theta,stan::math::default_policy());
    }

      // Binomial CDF
      template <bool propto, typename T_n, typename T_N, typename T_prob, class Policy>
      typename return_type<T_prob>::type
      binomial_cdf(const T_n& n, const T_N& N, const T_prob& theta, const Policy&) {
          
          static const char* function = "stan::prob::binomial_cdf(%1%)";
          
          using stan::math::check_finite;
          using stan::math::check_bounded;
          using stan::math::check_nonnegative;
          using stan::math::value_of;
          using stan::math::check_consistent_sizes;
          using stan::prob::include_summand;
          
          // Ensure non-zero arguments lenghts
          if (!(stan::length(n) && stan::length(N) && stan::length(theta)))
              return 0.0;
          
          double P(1.0);
          
          // Validate arguments
          if (!check_bounded(function, n, 0, N, "Successes variable", &P, Policy()))
              return P;
          
          if (!check_nonnegative(function, N, "Population size parameter", &P, Policy()))
              return P;
          
          if (!check_finite(function, theta, "Probability parameter", &P, Policy()))
              return P;
          
          if (!check_bounded(function, theta, 0.0, 1.0, "Probability parameter", &P, Policy()))
              return P;
          
          if (!(check_consistent_sizes(function, n, N,theta, 
                                       "Successes variable","Population size parameter","Probability parameter",
                                       &P, Policy())))
              return P;
          
          
          // Return if everything is constant and only proportionality is required
          if (!include_summand<propto,T_prob>::value)
              return 0.0;
          
          // Wrap arguments in vector views
          VectorView<const T_n> n_vec(n);
          VectorView<const T_N> N_vec(N);
          VectorView<const T_prob> theta_vec(theta);
          size_t size = max_size(n, N, theta);
          
          // Compute vectorized CDF and gradient
          using stan::math::value_of;
          using boost::math::ibetac;
          using boost::math::ibeta_derivative;
          
          agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);
          
          std::fill(operands_and_partials.all_partials,
                    operands_and_partials.all_partials + operands_and_partials.nvaris, 0.0);
          
          for (size_t i = 0; i < size; i++) {
              
              const double n_dbl = value_of(n_vec[i]);
              const double N_dbl = value_of(N_vec[i]);
              const double theta_dbl = value_of(theta_vec[i]);
              
              const double Pi = ibetac(n_dbl, N_dbl, theta_dbl);
  
              P *= Pi;
              
              if (!is_constant_struct<T_prob>::value)
                  operands_and_partials.d_x1[i] += - ibeta_derivative(n_dbl, N_dbl, theta_dbl) / Pi;
              
          }
          
          for (size_t i = 0; i < size; i++) {
              
              if (!is_constant_struct<T_prob>::value)
                  operands_and_partials.d_x1[i] *= P;
              
          }
          
          return P;
      }
      
      template <bool propto, typename T_n, typename T_N, typename T_prob>
      inline typename return_type<T_prob>::type
      binomial_cdf(const T_n& n, const T_N& N, const T_prob& theta) {
          return binomial_cdf<propto>(n, N, theta,stan::math::default_policy());
      }
      
      
      template <typename T_n, typename T_N, typename T_prob, class Policy>
      inline typename return_type<T_prob>::type
      binomial_cdf(const T_n& n, const T_N& N, const T_prob& theta, const Policy&) {
          return binomial_cdf<false>(n, N, theta,Policy());
      }
      
      template <typename T_n, typename T_N, typename T_prob>
      inline typename return_type<T_prob>::type
      binomial_cdf(const T_n& n, const T_N& N, const T_prob& theta) {
          return binomial_cdf<false>(n, N, theta, stan::math::default_policy());
      }

    
  }
}
#endif
