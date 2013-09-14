#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__POISSON_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__POISSON_HPP__

#include <boost/random/poisson_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <limits>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Poisson(n|lambda)  [lambda > 0;  n >= 0]
    template <bool propto,
              typename T_n, typename T_rate>
    typename return_type<T_rate>::type
    poisson_log(const T_n& n, const T_rate& lambda) {

      static const char* function = "stan::prob::poisson_log(%1%)";
      
      using boost::math::lgamma;
      using stan::math::check_consistent_sizes;
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::prob::include_summand;
      using stan::math::value_of;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(lambda)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args
      if (!check_nonnegative(function, n, "Random variable", &logp))
        return logp;
      if (!check_not_nan(function, lambda,
                         "Rate parameter", &logp))
        return logp;
      if (!check_nonnegative(function, lambda,
                             "Rate parameter", &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,lambda,
                                   "Random variable","Rate parameter",
                                   &logp)))
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
          operands_and_partials.d_x1[i] 
            += n_vec[i] / value_of(lambda_vec[i]) - 1.0;
        
      }


      return operands_and_partials.to_var(logp);
    }
    
    template <typename T_n,
              typename T_rate>
    inline
    typename return_type<T_rate>::type
    poisson_log(const T_n& n, const T_rate& lambda) {
      return poisson_log<false>(n,lambda);
    }

    // PoissonLog(n|alpha)  [n >= 0]   = Poisson(n|exp(alpha))
    template <bool propto,
              typename T_n, typename T_log_rate>
    typename return_type<T_log_rate>::type
    poisson_log_log(const T_n& n, const T_log_rate& alpha) {

      static const char* function = "stan::prob::poisson_log_log(%1%)";
      
      using boost::math::lgamma;
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      using std::exp;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(alpha)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args
      if (!check_nonnegative(function, n, "Random variable", &logp))
        return logp;
      if (!check_not_nan(function, alpha,
                         "Log rate parameter", &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,alpha,
                                   "Random variable","Log rate parameter",
                                   &logp)))
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
      DoubleVectorView<include_summand<propto,T_log_rate>::value,
        is_vector<T_log_rate>::value>
        exp_alpha(length(alpha));
      for (size_t i = 0; i < length(alpha); i++)
        if (include_summand<propto,T_log_rate>::value)
          exp_alpha[i] = exp(value_of(alpha_vec[i]));
      using stan::math::multiply_log;
      for (size_t i = 0; i < size; i++) {
        if (!(alpha_vec[i] == -std::numeric_limits<double>::infinity() 
              && n_vec[i] == 0)) {
          if (include_summand<propto>::value)
            logp -= lgamma(n_vec[i] + 1.0);
          if (include_summand<propto,T_log_rate>::value)
            logp += n_vec[i] * value_of(alpha_vec[i]) - exp_alpha[i];
        }

        // gradients
        if (!is_constant_struct<T_log_rate>::value)
          operands_and_partials.d_x1[i] += n_vec[i] - exp_alpha[i];
      }
      return operands_and_partials.to_var(logp);
    }
    
    template <typename T_n,
              typename T_log_rate>
    inline
    typename return_type<T_log_rate>::type
    poisson_log_log(const T_n& n, const T_log_rate& alpha) {
      return poisson_log_log<false>(n,alpha);
    }

    // Poisson CDF
    template <typename T_n, typename T_rate>
    typename return_type<T_rate>::type
    poisson_cdf(const T_n& n, const T_rate& lambda) {
      static const char* function = "stan::prob::poisson_cdf(%1%)";
          
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
          
      // Ensure non-zero argument slengths
      if (!(stan::length(n) && stan::length(lambda))) 
        return 1.0;
          
      double P(1.0);
          
      // Validate arguments
      if (!check_not_nan(function, lambda, "Rate parameter", &P))
        return P;
      if (!check_nonnegative(function, lambda, "Rate parameter", &P))
        return P;
      if (!(check_consistent_sizes(function, n,lambda,
                                   "Random variable","Rate parameter",
                                   &P)))
        return P;
          
      // Wrap arguments into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_rate> lambda_vec(lambda);
      size_t size = max_size(n, lambda);
          
      // Compute vectorized CDF and gradient
      using stan::math::value_of;
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_q;
          
      agrad::OperandsAndPartials<T_rate> operands_and_partials(lambda);

      std::fill(operands_and_partials.all_partials,
                operands_and_partials.all_partials 
                + operands_and_partials.nvaris, 0.0);
        
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0) 
          return operands_and_partials.to_var(0.0);
      }
        
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) == std::numeric_limits<double>::infinity())
          continue;
          
        const double n_dbl = value_of(n_vec[i]);
        const double lambda_dbl = value_of(lambda_vec[i]);
        const double Pi = gamma_q(n_dbl+1, lambda_dbl);

        P *= Pi;
  
        if (!is_constant_struct<T_rate>::value)
          operands_and_partials.d_x1[i] 
            -= gamma_p_derivative(n_dbl + 1, lambda_dbl) / Pi;
      }
      
      if (!is_constant_struct<T_rate>::value)
        for(size_t i = 0; i < stan::length(lambda); ++i) 
          operands_and_partials.d_x1[i] *= P;
      
      return operands_and_partials.to_var(P);
    }

    template <typename T_n, typename T_rate>
    typename return_type<T_rate>::type
    poisson_cdf_log(const T_n& n, const T_rate& lambda) {
      static const char* function = "stan::prob::poisson_cdf_log(%1%)";
          
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
          
      // Ensure non-zero argument slengths
      if (!(stan::length(n) && stan::length(lambda))) 
        return 0.0;
          
      double P(0.0);
          
      // Validate arguments
      if (!check_not_nan(function, lambda, "Rate parameter", &P))
        return P;
      if (!check_nonnegative(function, lambda, "Rate parameter", &P))
        return P;
      if (!(check_consistent_sizes(function, n,lambda,
                                   "Random variable","Rate parameter",
                                   &P)))
        return P;
          
      // Wrap arguments into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_rate> lambda_vec(lambda);
      size_t size = max_size(n, lambda);
          
      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_q;
          
      agrad::OperandsAndPartials<T_rate> operands_and_partials(lambda);

      std::fill(operands_and_partials.all_partials,
                operands_and_partials.all_partials 
                + operands_and_partials.nvaris, 0.0);
        
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as neg infinity
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
        
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) == std::numeric_limits<double>::infinity())
          continue;
          
        const double n_dbl = value_of(n_vec[i]);
        const double lambda_dbl = value_of(lambda_vec[i]);
        const double Pi = gamma_q(n_dbl+1, lambda_dbl);

        P += log(Pi);
  
        if (!is_constant_struct<T_rate>::value)
          operands_and_partials.d_x1[i] 
            -= gamma_p_derivative(n_dbl + 1, lambda_dbl) / Pi;
      }
      
      return operands_and_partials.to_var(P);
    }

    template <typename T_n, typename T_rate>
    typename return_type<T_rate>::type
    poisson_ccdf_log(const T_n& n, const T_rate& lambda) {
      static const char* function = "stan::prob::poisson_ccdf_log(%1%)";
          
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
          
      // Ensure non-zero argument slengths
      if (!(stan::length(n) && stan::length(lambda))) 
        return 0.0;
          
      double P(0.0);
          
      // Validate arguments
      if (!check_not_nan(function, lambda, "Rate parameter", &P))
        return P;
      if (!check_nonnegative(function, lambda, "Rate parameter", &P))
        return P;
      if (!(check_consistent_sizes(function, n,lambda,
                                   "Random variable","Rate parameter",
                                   &P)))
        return P;
          
      // Wrap arguments into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_rate> lambda_vec(lambda);
      size_t size = max_size(n, lambda);
          
      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_q;
          
      agrad::OperandsAndPartials<T_rate> operands_and_partials(lambda);

      std::fill(operands_and_partials.all_partials,
                operands_and_partials.all_partials 
                + operands_and_partials.nvaris, 0.0);
        
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as neg infinity
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0) 
          return operands_and_partials.to_var(0.0);
      }
        
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) == std::numeric_limits<double>::infinity())
          return operands_and_partials.to_var(stan::math::negative_infinity());
          
        const double n_dbl = value_of(n_vec[i]);
        const double lambda_dbl = value_of(lambda_vec[i]);
        const double Pi = 1.0 - gamma_q(n_dbl+1, lambda_dbl);

        P += log(Pi);
  
        if (!is_constant_struct<T_rate>::value)
          operands_and_partials.d_x1[i] 
            += gamma_p_derivative(n_dbl + 1, lambda_dbl) / Pi;
      }
      
      return operands_and_partials.to_var(P);
    }

    template <class RNG>
    inline int
    poisson_rng(const double lambda,
                RNG& rng) {
      using boost::variate_generator;
      using boost::random::poisson_distribution;

      static const char* function = "stan::prob::poisson_rng(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
 
      if (!check_not_nan(function, lambda,
                         "Rate parameter"))
        return 0;
      if (!check_nonnegative(function, lambda,
                             "Rate parameter"))
        return 0;

      variate_generator<RNG&, poisson_distribution<> >
        poisson_rng(rng, poisson_distribution<>(lambda));
      return poisson_rng();
    }
      
  }
}
#endif
