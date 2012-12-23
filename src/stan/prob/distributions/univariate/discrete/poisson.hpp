#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__POISSON_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__POISSON_HPP__

#include <limits>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Poisson(n|lambda)  [lambda > 0;  n >= 0]
    template <bool propto,
              typename T_n, typename T_rate, 
              class Policy>
    typename return_type<T_rate>::type
    poisson_log(const T_n& n, const T_rate& lambda, 
                const Policy&) {

      static const char* function = "stan::prob::poisson_log(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(lambda)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args
      if (!check_nonnegative(function, n, "Random variable", &logp, Policy()))
        return logp;
      if (!check_not_nan(function, lambda,
                         "Rate parameter", &logp, Policy()))
        return logp;
      if (!check_nonnegative(function, lambda,
                             "Rate parameter", &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,lambda,
                                   "Random variable","Rate parameter",
                                   &logp, Policy())))
        return logp;
      
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_rate>::value)
        return 0.0;

      // set up expression templates wrapping scalars/vecs into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_rate> lambda_vec(lambda);
      size_t size = max_size(n, lambda);

      for (size_t i = 0; i < size; i++)
        if (std::isinf(lambda_vec[i]))
          return LOG_ZERO;
      for (size_t i = 0; i < size; i++)
        if (lambda_vec[i] == 0 && n_vec[i] != 0)
          return LOG_ZERO;
      
      // return accumulator with gradients
      agrad::OperandsAndPartials<T_rate> operands_and_partials(lambda);

      using stan::math::multiply_log;
      for (size_t i = 0; i < size; i++) {
        if (!(lambda_vec[i] == 0 && n_vec[i] == 0)) {
          if (include_summand<propto>::value)
            logp -= lgamma(n_vec[i] + 1.0);
          if (include_summand<propto,T_rate>::value)
            logp += multiply_log(n_vec[i], value_of(lambda_vec[i])) 
              - value_of(lambda_vec[i]);
        }

        // gradients
        if (!is_constant_struct<T_rate>::value)
          operands_and_partials.d_x1[i] += n_vec[i] / value_of(lambda_vec[i]) - 1.0;
        
      }


      return operands_and_partials.to_var(logp);
    }
    
    template <bool propto,
              typename T_n,
              typename T_rate>
    inline
    typename return_type<T_rate>::type
    poisson_log(const T_n& n, const T_rate& lambda) {
      return poisson_log<propto>(n,lambda,stan::math::default_policy());
    }


    template <typename T_n,
              typename T_rate, 
              class Policy>
    inline
    typename return_type<T_rate>::type
    poisson_log(const T_n& n, const T_rate& lambda, 
                const Policy&) {
      return poisson_log<false>(n,lambda,Policy());
    }


    template <typename T_n,
              typename T_rate>
    inline
    typename return_type<T_rate>::type
    poisson_log(const T_n& n, const T_rate& lambda) {
      return poisson_log<false>(n,lambda,stan::math::default_policy());
    }





    // PoissonLog(n|alpha)  [n >= 0]   = Poisson(n|exp(alpha))
    template <bool propto,
              typename T_n, typename T_log_rate, 
              class Policy>
    typename return_type<T_log_rate>::type
    poisson_log_log(const T_n& n, const T_log_rate& alpha, 
                const Policy&) {

      static const char* function = "stan::prob::poisson_log_log(%1%)";
      
      using std::exp;
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(alpha)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args
      if (!check_nonnegative(function, n, "Random variable", &logp, Policy()))
        return logp;
      if (!check_not_nan(function, alpha,
                         "Log rate parameter", &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,alpha,
                                   "Random variable","Log rate parameter",
                                   &logp, Policy())))
        return logp;
      
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_log_rate>::value)
        return 0.0;

      // set up expression templates wrapping scalars/vecs into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_log_rate> alpha_vec(alpha);
      size_t size = max_size(n, alpha);

      // FIXME: first loop size of alpha_vec, second loop if-ed for size==1
      for (size_t i = 0; i < size; i++)
        if (std::numeric_limits<double>::infinity() == alpha_vec[i])
          return LOG_ZERO;
      for (size_t i = 0; i < size; i++)
        if (-std::numeric_limits<double>::infinity() == alpha_vec[i] 
            && n_vec[i] != 0)
          return LOG_ZERO;
      
      // return accumulator with gradients
      agrad::OperandsAndPartials<T_log_rate> operands_and_partials(alpha);

      // FIXME: cache value_of for alpha_vec?  faster if only one?
      
      using stan::math::multiply_log;
      for (size_t i = 0; i < size; i++) {
        if (!(alpha_vec[i] == -std::numeric_limits<double>::infinity() 
              && n_vec[i] == 0)) {
          if (include_summand<propto>::value)
            logp -= lgamma(n_vec[i] + 1.0);
          if (include_summand<propto,T_log_rate>::value)
            logp += n_vec[i] * value_of(alpha_vec[i]) - exp(value_of(alpha_vec[i]));
        }

        // gradients
        if (!is_constant_struct<T_log_rate>::value)
          operands_and_partials.d_x1[i] += n_vec[i] - exp(value_of(alpha_vec[i]));
      }
      return operands_and_partials.to_var(logp);
    }
    
    template <bool propto,
              typename T_n,
              typename T_log_rate>
    inline
    typename return_type<T_log_rate>::type
    poisson_log_log(const T_n& n, const T_log_rate& alpha) {
      return poisson_log_log<propto>(n,alpha,stan::math::default_policy());
    }


    template <typename T_n,
              typename T_log_rate, 
              class Policy>
    inline
    typename return_type<T_log_rate>::type
    poisson_log_log(const T_n& n, const T_log_rate& alpha, 
                const Policy&) {
      return poisson_log_log<false>(n,alpha,Policy());
    }


    template <typename T_n,
              typename T_log_rate>
    inline
    typename return_type<T_log_rate>::type
    poisson_log_log(const T_n& n, const T_log_rate& alpha) {
      return poisson_log_log<false>(n,alpha,stan::math::default_policy());
    }

      // Poisson CDF
      template <bool propto, typename T_n, typename T_rate, class Policy>
      typename return_type<T_rate>::type
      poisson_cdf(const T_n& n, const T_rate& lambda, const Policy&) {
          
          static const char* function = "stan::prob::poisson_cdf(%1%)";
          
          using stan::math::check_not_nan;
          using stan::math::check_nonnegative;
          using stan::math::value_of;
          using stan::math::check_consistent_sizes;
          using stan::prob::include_summand;
          
          // Ensure non-zero argument slengths
          if (!(stan::length(n) && stan::length(lambda))) 
              return 0.0;
          
          double P(1.0);
          
          // Validate arguments
          if (!check_nonnegative(function, n, "Random variable", &P, Policy()))
              return P;
          
          if (!check_not_nan(function, lambda, "Rate parameter", &P, Policy()))
              return P;
          
          if (!check_nonnegative(function, lambda, "Rate parameter", &P, Policy()))
              return P;
          
          if (!(check_consistent_sizes(function, n,lambda,
                                       "Random variable","Rate parameter",
                                       &P, Policy())))
              return P;
          
          // Return if everything is constant and only proportionality is required
          if (!include_summand<propto,T_rate>::value)
              return 0.0;
          
          // Wrap arguments into vector views
          VectorView<const T_n> n_vec(n);
          VectorView<const T_rate> lambda_vec(lambda);
          size_t size = max_size(n, lambda);
          
          // Validate vector views - redundant with the above validations?
          for (size_t i = 0; i < size; i++)
              if (std::isinf(lambda_vec[i])) return LOG_ZERO;
          
          for (size_t i = 0; i < size; i++)
              if (lambda_vec[i] == 0 && n_vec[i] != 0)
                  return LOG_ZERO;
          

          // Compute vectorized CDF and gradient
          using stan::math::value_of;
          using boost::math::gamma_p_derivative;
          using boost::math::gamma_p;
          
          agrad::OperandsAndPartials<T_rate> operands_and_partials(lambda);
          
          std::fill(operands_and_partials.all_partials,
                    operands_and_partials.all_partials + operands_and_partials.nvaris, 0.0);
          
          for (size_t i = 0; i < size; i++) 
          {

              const double n_dbl = value_of(n_vec[i]);
              const double lambda_dbl = value_of(lambda_vec[i]);
              
              const double Pi = gamma_p(n_dbl, lambda_dbl);
              
              P *= Pi;
              
              if (!is_constant_struct<T_rate>::value)
                  operands_and_partials.d_x1[i] += gamma_p_derivative(n_dbl, lambda_dbl) / Pi;
              
          }
          
          for (size_t i = 0; i < size; i++) {
              
              if (!is_constant_struct<T_rate>::value)
                  operands_and_partials.d_x1[n] *= P;
              
          }
          
          return operands_and_partials.to_var(P);
      }
      
      template <bool propto, typename T_n, typename T_rate>
      inline typename return_type<T_rate>::type
      poisson_cdf(const T_n& n, const T_rate& lambda) {
          return poisson_cdf<propto>(n, lambda, stan::math::default_policy());
      }
      
      
      template <typename T_n, typename T_rate, class Policy>
      inline typename return_type<T_rate>::type
      poisson_cdf(const T_n& n, const T_rate& lambda, const Policy&) {
          return poisson_cdf<false>(n, lambda, Policy());
      }
      
      
      template <typename T_n, typename T_rate>
      inline typename return_type<T_rate>::type
      poisson_cdf(const T_n& n, const T_rate& lambda) {
          return poisson_cdf<false>(n, lambda, stan::math::default_policy());
      }

      
  }
}
#endif
