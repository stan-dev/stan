#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BETA_BINOMIAL_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BETA_BINOMIAL_HPP

#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>
#include <stan/math/functions/lbeta.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>
#include <stan/prob/distributions/univariate/continuous/beta.hpp>
#include <stan/prob/internal_math.hpp>


namespace stan {
  
  namespace prob {

    // BetaBinomial(n|alpha,beta) [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto,
              typename T_n, typename T_N,
              typename T_size1, typename T_size2>
    typename return_type<T_size1,T_size2>::type
    beta_binomial_log(const T_n& n, 
                      const T_N& N, 
                      const T_size1& alpha, 
                      const T_size2& beta) {
      static const std::string function("stan::prob::beta_binomial_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(N)
            && stan::length(alpha)
            && stan::length(beta)))
        return 0.0;
      
      double logp(0.0);
      check_nonnegative(function, "Population size parameter", N);
      check_positive_finite(function, "First prior sample size parameter", alpha);
      check_positive_finite(function, "Second prior sample size parameter", beta);
      check_consistent_sizes(function,
                             "Successes variable", n,
                             "Population size parameter", N,
                             "First prior sample size parameter", alpha,
                             "Second prior sample size parameter", beta);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_size1,T_size2>::value)
        return 0.0;

      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_size1> alpha_vec(alpha);
      VectorView<const T_size2> beta_vec(beta);
      size_t size = max_size(n, N, alpha, beta);
      
      for (size_t i = 0; i < size; i++) {
        if (n_vec[i] < 0 || n_vec[i] > N_vec[i])
          return LOG_ZERO;
      }
      
      using stan::math::lbeta;
      using stan::math::binomial_coefficient_log;
      using boost::math::digamma;

      DoubleVectorView<include_summand<propto>::value,
                       is_vector<T_n>::value || is_vector<T_N>::value> 
        normalizing_constant(max_size(N,n));
      for (size_t i = 0; i < max_size(N,n); i++)
        if (include_summand<propto>::value)
          normalizing_constant[i] 
            = binomial_coefficient_log(N_vec[i],n_vec[i]);
      
      DoubleVectorView<include_summand<propto,T_size1,T_size2>::value,
                       is_vector<T_n>::value || is_vector<T_N>::value 
                       || is_vector<T_size1>::value || is_vector<T_size2>::value>
        lbeta_numerator(size);
      for (size_t i = 0; i < size; i++)
        if (include_summand<propto,T_size1,T_size2>::value)
          lbeta_numerator[i] = lbeta(n_vec[i] + value_of(alpha_vec[i]),
                                     N_vec[i] - n_vec[i] 
                                     + value_of(beta_vec[i]));
      DoubleVectorView<include_summand<propto,T_size1,T_size2>::value,
                       is_vector<T_size1>::value || is_vector<T_size2>::value>
        lbeta_denominator(max_size(alpha,beta));
      for (size_t i = 0; i < max_size(alpha,beta); i++)
        if (include_summand<propto,T_size1,T_size2>::value)
          lbeta_denominator[i] = lbeta(value_of(alpha_vec[i]), 
                                       value_of(beta_vec[i]));
      
      DoubleVectorView<!is_constant_struct<T_size1>::value,
                       is_vector<T_n>::value || is_vector<T_size1>::value> 
        digamma_n_plus_alpha(max_size(n,alpha));
      for (size_t i = 0; i < max_size(n,alpha); i++)
        if (!is_constant_struct<T_size1>::value)
          digamma_n_plus_alpha[i] 
            = digamma(n_vec[i] + value_of(alpha_vec[i]));

      DoubleVectorView<!is_constant_struct<T_size1>::value
                       || !is_constant_struct<T_size2>::value,
                       is_vector<T_N>::value 
                       || is_vector<T_size1>::value 
                       || is_vector<T_size1>::value> 
        digamma_N_plus_alpha_plus_beta(max_size(N,alpha,beta));
      for (size_t i = 0; i < max_size(N,alpha,beta); i++)
        if (!is_constant_struct<T_size1>::value 
            || !is_constant_struct<T_size2>::value)
          digamma_N_plus_alpha_plus_beta[i] 
            = digamma(N_vec[i] + value_of(alpha_vec[i]) + value_of(beta_vec[i]));

      DoubleVectorView<!is_constant_struct<T_size1>::value
                       || !is_constant_struct<T_size2>::value,
                       is_vector<T_size1>::value
                       || is_vector<T_size1>::value> 
        digamma_alpha_plus_beta(max_size(alpha,beta));
      for (size_t i = 0; i < max_size(alpha,beta); i++)
        if (!is_constant_struct<T_size1>::value 
            || !is_constant_struct<T_size2>::value)
          digamma_alpha_plus_beta[i] 
            = digamma(value_of(alpha_vec[i]) + value_of(beta_vec[i]));

      DoubleVectorView<!is_constant_struct<T_size1>::value, 
                       is_vector<T_size1>::value>
        digamma_alpha(length(alpha));
      for (size_t i = 0; i < length(alpha); i++)
        if (!is_constant_struct<T_size1>::value)
          digamma_alpha[i] = digamma(value_of(alpha_vec[i]));

      DoubleVectorView<!is_constant_struct<T_size2>::value, 
                       is_vector<T_size2>::value>
        digamma_beta(length(beta));
      for (size_t i = 0; i < length(beta); i++)
        if (!is_constant_struct<T_size2>::value)
          digamma_beta[i] = digamma(value_of(beta_vec[i]));

      agrad::OperandsAndPartials<T_n,T_N,T_size1,T_size2> 
        operands_and_partials(n,N,alpha,beta);
      for (size_t i = 0; i < size; i++) {
        if (include_summand<propto>::value)
          logp += normalizing_constant[i];
        if (include_summand<propto,T_size1,T_size2>::value)
          logp += lbeta_numerator[i] 
            - lbeta_denominator[i];
        
        if (!is_constant_struct<T_size1>::value)
          operands_and_partials.d_x3[i] 
            += digamma_n_plus_alpha[i]
            - digamma_N_plus_alpha_plus_beta[i]
            + digamma_alpha_plus_beta[i]
            - digamma_alpha[i];
        if (!is_constant_struct<T_size2>::value)
          operands_and_partials.d_x4[i] 
            += digamma(value_of(N_vec[i]-n_vec[i]+beta_vec[i]))
            - digamma_N_plus_alpha_plus_beta[i]
            + digamma_alpha_plus_beta[i]
            - digamma_beta[i];
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_n,
              typename T_N,
              typename T_size1,
              typename T_size2>
    typename return_type<T_size1,T_size2>::type
    beta_binomial_log(const T_n& n, const T_N& N, 
                      const T_size1& alpha, const T_size2& beta) {
      return beta_binomial_log<false>(n,N,alpha,beta);
    }

    // Beta-Binomial CDF
    template <typename T_n, typename T_N, 
              typename T_size1, typename T_size2>
    typename return_type<T_size1,T_size2>::type
    beta_binomial_cdf(const T_n& n, const T_N& N, const T_size1& alpha, 
                      const T_size2& beta) {
          
      static const std::string function("stan::prob::beta_binomial_cdf");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero argument lengths
      if (!(stan::length(n) && stan::length(N) && stan::length(alpha) 
            && stan::length(beta)))
        return 1.0;
          
      double P(1.0);
          
      // Validate arguments
      check_nonnegative(function, "Population size parameter", N);
      check_positive_finite(function, "First prior sample size parameter", alpha);
      check_positive_finite(function, "Second prior sample size parameter", beta);
      check_consistent_sizes(function,                           
                             "Successes variable", n, 
                             "Population size parameter", N, 
                             "First prior sample size parameter", alpha, 
                             "Second prior sample size parameter", beta);

      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_size1> alpha_vec(alpha);
      VectorView<const T_size2> beta_vec(beta);
      size_t size = max_size(n, N, alpha, beta);
          
      // Compute vectorized CDF and gradient
      using boost::math::lgamma;
      using boost::math::digamma;

      agrad::OperandsAndPartials<T_size1, T_size2> 
        operands_and_partials(alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0) 
          return operands_and_partials.to_var(0.0);
      }
          
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= value_of(N_vec[i])) {
          continue;
        }
              
        const double n_dbl = value_of(n_vec[i]);
        const double N_dbl = value_of(N_vec[i]);
        const double alpha_dbl = value_of(alpha_vec[i]);
        const double beta_dbl = value_of(beta_vec[i]);
              
        const double mu = alpha_dbl + n_dbl + 1;
        const double nu = beta_dbl + N_dbl - n_dbl - 1;
              
        const double F = stan::math::F32(1, mu, -N_dbl + n_dbl + 1, n_dbl + 2, 
                                         1 - nu, 1);
              
        double C = lgamma(nu) - lgamma(N_dbl - n_dbl);
        C += lgamma(mu) - lgamma(n_dbl + 2);
        C += lgamma(N_dbl + 2) - lgamma(N_dbl + alpha_dbl + beta_dbl);
        C = std::exp(C);
                
        C *= F / boost::math::beta(alpha_dbl, beta_dbl);
        C /= N_dbl + 1;
              
        const double Pi = 1 - C;
              
        P *= Pi;
              
        double dF[6];
        double digammaOne = 0;
        double digammaTwo = 0;
              
        if ( (!is_constant_struct<T_size1>::value) 
             || (!is_constant_struct<T_size2>::value) ) {
          digammaOne = digamma(mu + nu);
          digammaTwo = digamma(alpha_dbl + beta_dbl);
          stan::math::gradF32(dF, 1, mu, -N_dbl + n_dbl + 1, n_dbl + 2,
                              1 - nu, 1);
        }
        if (!is_constant_struct<T_size1>::value) {
          const double g 
            = - C * (digamma(mu) - digammaOne + dF[1] / F
                     - digamma(alpha_dbl) + digammaTwo);
          operands_and_partials.d_x1[i] 
            += g / Pi;
        }
        if (!is_constant_struct<T_size2>::value) {
          const double g 
            = - C * (digamma(nu) - digammaOne - dF[4] / F - digamma(beta_dbl) 
                     + digammaTwo);
          operands_and_partials.d_x2[i] 
            += g / Pi;
        }
      }
          
      if (!is_constant_struct<T_size1>::value)
        for(size_t i = 0; i < stan::length(alpha); ++i)
          operands_and_partials.d_x1[i] *= P;
      if (!is_constant_struct<T_size2>::value)
        for(size_t i = 0; i < stan::length(beta); ++i)
          operands_and_partials.d_x2[i] *= P;
          
      return operands_and_partials.to_var(P);
    }

    template <typename T_n, typename T_N, 
              typename T_size1, typename T_size2>
    typename return_type<T_size1,T_size2>::type
    beta_binomial_cdf_log(const T_n& n, const T_N& N, const T_size1& alpha, 
                          const T_size2& beta) {
          
      static const std::string function("stan::prob::beta_binomial_cdf_log");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero argument lengths
      if (!(stan::length(n) && stan::length(N) && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;
          
      double P(0.0);
          
      // Validate arguments
      check_nonnegative(function, "Population size parameter", N);
      check_positive_finite(function, "First prior sample size parameter", alpha);
      check_positive_finite(function, "Second prior sample size parameter", beta);
      check_consistent_sizes(function,                           
                             "Successes variable", n, 
                             "Population size parameter", N, 
                             "First prior sample size parameter", alpha, 
                             "Second prior sample size parameter", beta);

      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_size1> alpha_vec(alpha);
      VectorView<const T_size2> beta_vec(beta);
      size_t size = max_size(n, N, alpha, beta);
          
      // Compute vectorized cdf_log and gradient
      using boost::math::lgamma;
      using boost::math::digamma;

      agrad::OperandsAndPartials<T_size1, T_size2> 
        operands_and_partials(alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as neg infinity
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
          
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= value_of(N_vec[i])) {
          continue;
        }
              
        const double n_dbl = value_of(n_vec[i]);
        const double N_dbl = value_of(N_vec[i]);
        const double alpha_dbl = value_of(alpha_vec[i]);
        const double beta_dbl = value_of(beta_vec[i]);
              
        const double mu = alpha_dbl + n_dbl + 1;
        const double nu = beta_dbl + N_dbl - n_dbl - 1;
              
        const double F = stan::math::F32(1, mu, -N_dbl + n_dbl + 1, n_dbl + 2, 
                                         1 - nu, 1);
              
        double C = lgamma(nu) - lgamma(N_dbl - n_dbl);
        C += lgamma(mu) - lgamma(n_dbl + 2);
        C += lgamma(N_dbl + 2) - lgamma(N_dbl + alpha_dbl + beta_dbl);
        C = std::exp(C);
                
        C *= F / boost::math::beta(alpha_dbl, beta_dbl);
        C /= N_dbl + 1;
              
        const double Pi = 1 - C;
              
        P += log(Pi);
              
        double dF[6];
        double digammaOne = 0;
        double digammaTwo = 0;
              
        if ( (!is_constant_struct<T_size1>::value) 
             || (!is_constant_struct<T_size2>::value) ) {
          digammaOne = digamma(mu + nu);
          digammaTwo = digamma(alpha_dbl + beta_dbl);
          stan::math::gradF32(dF, 1, mu, -N_dbl + n_dbl + 1, n_dbl + 2,
                              1 - nu, 1);
        }
        if (!is_constant_struct<T_size1>::value) {
          const double g 
            = - C * (digamma(mu) - digammaOne + dF[1] / F
                     - digamma(alpha_dbl) + digammaTwo);
          operands_and_partials.d_x1[i] += g / Pi;
        }
        if (!is_constant_struct<T_size2>::value) {
          const double g 
            = - C * (digamma(nu) - digammaOne - dF[4] / F - digamma(beta_dbl) 
                     + digammaTwo);
          operands_and_partials.d_x2[i] += g / Pi;
        }
      }
          
      return operands_and_partials.to_var(P);
    }
      
    template <typename T_n, typename T_N, 
              typename T_size1, typename T_size2>
    typename return_type<T_size1,T_size2>::type
    beta_binomial_ccdf_log(const T_n& n, const T_N& N, const T_size1& alpha, 
                           const T_size2& beta) {
          
      static const std::string function("stan::prob::beta_binomial_ccdf_log");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero argument lengths
      if (!(stan::length(n) && stan::length(N) && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;
          
      double P(0.0);
          
      // Validate arguments
      check_nonnegative(function, "Population size parameter", N);
      check_positive_finite(function, "First prior sample size parameter", alpha);
      check_positive_finite(function, "Second prior sample size parameter", beta);
      check_consistent_sizes(function,
                             "Successes variable", n, 
                             "Population size parameter", N, 
                             "First prior sample size parameter", alpha, 
                             "Second prior sample size parameter", beta);

      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_size1> alpha_vec(alpha);
      VectorView<const T_size2> beta_vec(beta);
      size_t size = max_size(n, N, alpha, beta);
          
      // Compute vectorized cdf_log and gradient
      using boost::math::lgamma;
      using boost::math::digamma;

      agrad::OperandsAndPartials<T_size1, T_size2> 
        operands_and_partials(alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as neg infinity
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0) 
          return operands_and_partials.to_var(0.0);
      }
          
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= value_of(N_vec[i])) {
          return operands_and_partials.to_var(stan::math::negative_infinity());
        }
              
        const double n_dbl = value_of(n_vec[i]);
        const double N_dbl = value_of(N_vec[i]);
        const double alpha_dbl = value_of(alpha_vec[i]);
        const double beta_dbl = value_of(beta_vec[i]);
              
        const double mu = alpha_dbl + n_dbl + 1;
        const double nu = beta_dbl + N_dbl - n_dbl - 1;
              
        const double F = stan::math::F32(1, mu, -N_dbl + n_dbl + 1, n_dbl + 2, 
                                         1 - nu, 1);
              
        double C = lgamma(nu) - lgamma(N_dbl - n_dbl);
        C += lgamma(mu) - lgamma(n_dbl + 2);
        C += lgamma(N_dbl + 2) - lgamma(N_dbl + alpha_dbl + beta_dbl);
        C = std::exp(C);
                
        C *= F / boost::math::beta(alpha_dbl, beta_dbl);
        C /= N_dbl + 1;
              
        const double Pi = C;
              
        P += log(Pi);
              
        double dF[6];
        double digammaOne = 0;
        double digammaTwo = 0;
              
        if ( (!is_constant_struct<T_size1>::value) 
             || (!is_constant_struct<T_size2>::value) ) {
          digammaOne = digamma(mu + nu);
          digammaTwo = digamma(alpha_dbl + beta_dbl);
          stan::math::gradF32(dF, 1, mu, -N_dbl + n_dbl + 1, n_dbl + 2,
                              1 - nu, 1);
        }
        if (!is_constant_struct<T_size1>::value) {
          const double g 
            = - C * (digamma(mu) - digammaOne + dF[1] / F
                     - digamma(alpha_dbl) + digammaTwo);
          operands_and_partials.d_x1[i] -= g / Pi;
        }
        if (!is_constant_struct<T_size2>::value) {
          const double g 
            = - C * (digamma(nu) - digammaOne - dF[4] / F - digamma(beta_dbl) 
                     + digammaTwo);
          operands_and_partials.d_x2[i] -= g / Pi;
        }
      }
          
      return operands_and_partials.to_var(P);
    }

    template <class RNG>
    inline int
    beta_binomial_rng(const int N,
                      const double alpha,
                      const double beta,
                      RNG& rng) {

      static const std::string function("stan::prob::beta_binomial_rng");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
  
      check_nonnegative(function, "Population size parameter", N);
      check_positive_finite(function, "First prior sample size parameter", alpha);
      check_positive_finite(function, "Second prior sample size parameter", beta);
                            
      double a = stan::prob::beta_rng(alpha, beta, rng);
      while (a > 1 || a < 0) 
        a = stan::prob::beta_rng(alpha, beta, rng);
      return stan::prob::binomial_rng(N, a, rng);
    }
  }
}
#endif
