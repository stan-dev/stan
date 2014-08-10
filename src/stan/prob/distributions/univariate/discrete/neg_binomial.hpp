#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_HPP

#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/functions/digamma.hpp>
#include <stan/math/functions/lgamma.hpp>
#include <stan/math/functions/lbeta.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/continuous/gamma.hpp>
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

#include <stan/prob/internal_math/math/grad_reg_inc_beta.hpp>
#include <stan/prob/internal_math/math/inc_beta.hpp>
#include <stan/prob/internal_math/fwd/inc_beta.hpp>
#include <stan/prob/internal_math/rev/inc_beta.hpp>


namespace stan {

  namespace prob {

    // NegBinomial(n|alpha,beta)  [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto,
              typename T_n,
              typename T_shape, typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_log(const T_n& n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta) {
      typedef typename stan::partials_return_type<T_n,T_shape,
                                                  T_inv_scale>::type 
        T_partials_return;

      static const char* function = "stan::prob::neg_binomial_log(%1%)";

      using stan::math::check_positive_finite;      
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(alpha)
            && stan::length(beta)))
        return 0.0;
      
      T_partials_return logp(0.0);
      check_nonnegative(function, n, "Failures variable", &logp);
      check_positive_finite(function, alpha, "Shape parameter", &logp);
      check_positive_finite(function, beta, "Inverse scale parameter", &logp);
      check_consistent_sizes(function,
                             n,alpha,beta,
                             "Failures variable",
                             "Shape parameter","Inverse scale parameter",
                             &logp);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_shape,T_inv_scale>::value)
        return 0.0;

      using stan::math::multiply_log;
      using stan::math::binomial_coefficient_log;
      using stan::math::digamma;
      using stan::math::lgamma;
      using std::log;
      
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);

      agrad::OperandsAndPartials<T_shape,T_inv_scale> 
        operands_and_partials(alpha,beta);

      size_t len_ab = max_size(alpha,beta);
      VectorBuilder<true, T_partials_return, T_shape,T_inv_scale>
        lambda(len_ab);
      for (size_t i = 0; i < len_ab; ++i) 
        lambda[i] = value_of(alpha_vec[i]) / value_of(beta_vec[i]);

      VectorBuilder<true, T_partials_return, T_inv_scale>
        log1p_beta(length(beta));
      for (size_t i = 0; i < length(beta); ++i)
        log1p_beta[i] = log1p(value_of(beta_vec[i]));

      VectorBuilder<true, T_partials_return, T_inv_scale>
        log_beta_m_log1p_beta(length(beta));
      for (size_t i = 0; i < length(beta); ++i)
        log_beta_m_log1p_beta[i] = log(value_of(beta_vec[i])) - log1p_beta[i];

      VectorBuilder<true, T_partials_return, T_inv_scale,T_shape>
        alpha_times_log_beta_over_1p_beta(len_ab);
      for (size_t i = 0; i < len_ab; ++i)
        alpha_times_log_beta_over_1p_beta[i] 
          = value_of(alpha_vec[i])
          * log(value_of(beta_vec[i]) 
                / (1.0 + value_of(beta_vec[i])));

      VectorBuilder<!is_constant_struct<T_shape>::value, 
                    T_partials_return, T_shape>
        digamma_alpha(length(alpha));
      if (!is_constant_struct<T_shape>::value)
        for (size_t i = 0; i < length(alpha); ++i)
          digamma_alpha[i] = digamma(value_of(alpha_vec[i]));

      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_inv_scale> log_beta(length(beta));
      if (!is_constant_struct<T_shape>::value)
        for (size_t i = 0; i < length(beta); ++i)
          log_beta[i] = log(value_of(beta_vec[i]));

      VectorBuilder<!is_constant_struct<T_inv_scale>::value, 
                    T_partials_return, T_shape,T_inv_scale>
        lambda_m_alpha_over_1p_beta(len_ab);
      if (!is_constant_struct<T_inv_scale>::value)
        for (size_t i = 0; i < len_ab; ++i)
          lambda_m_alpha_over_1p_beta[i] =
            lambda[i]
            - ( value_of(alpha_vec[i]) 
                / (1.0 + value_of(beta_vec[i])) );

      for (size_t i = 0; i < size; i++) {
        if (alpha_vec[i] > 1e10) { // reduces numerically to Poisson
          if (include_summand<propto>::value)
            logp -= lgamma(n_vec[i] + 1.0);
          if (include_summand<propto,T_shape,T_inv_scale>::value)
            logp += multiply_log(n_vec[i], lambda[i]) - lambda[i];

          if (!is_constant_struct<T_shape>::value)
            operands_and_partials.d_x1[i]
              += n_vec[i] / value_of(alpha_vec[i]) 
              - 1.0 / value_of(beta_vec[i]);
          if (!is_constant_struct<T_inv_scale>::value)
            operands_and_partials.d_x2[i]
              += (lambda[i] - n_vec[i]) / value_of(beta_vec[i]) ;
        } else { // standard density definition
          if (include_summand<propto,T_shape>::value)
            if (n_vec[i] != 0)
              logp += binomial_coefficient_log(n_vec[i] 
                                               + value_of(alpha_vec[i])
                                               - 1.0, 
                                               n_vec[i]);
          if (include_summand<propto,T_shape,T_inv_scale>::value)
            logp += 
              alpha_times_log_beta_over_1p_beta[i] 
              - n_vec[i] * log1p_beta[i];

          if (!is_constant_struct<T_shape>::value)
            operands_and_partials.d_x1[i]
              += digamma(value_of(alpha_vec[i]) + n_vec[i])
              - digamma_alpha[i]
              + log_beta_m_log1p_beta[i];
          if (!is_constant_struct<T_inv_scale>::value)
            operands_and_partials.d_x2[i]
              += lambda_m_alpha_over_1p_beta[i]
              - n_vec[i]  / (value_of(beta_vec[i]) + 1.0);
        }
      }
      return operands_and_partials.to_var(logp,alpha,beta);
    }

    template <typename T_n, 
              typename T_shape, typename T_inv_scale>
    inline
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_log(const T_n& n, 
                     const T_shape& alpha, 
                     const T_inv_scale& beta) {
      return neg_binomial_log<false>(n,alpha,beta);
    }

    // Negative Binomial CDF
    template <typename T_n, typename T_shape, 
              typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_cdf(const T_n& n, const T_shape& alpha, 
                     const T_inv_scale& beta) {
      static const char* function = "stan::prob::neg_binomial_cdf(%1%)";
      typedef typename stan::partials_return_type<T_n,T_shape,
                                                  T_inv_scale>::type 
        T_partials_return;

      using stan::math::check_positive_finite;      
      using stan::math::check_nonnegative;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero arugment lengths
      if (!(stan::length(n) && stan::length(alpha) && stan::length(beta)))
        return 1.0;
          
      T_partials_return P(1.0);
          
      // Validate arguments
      check_positive_finite(function, alpha, "Shape parameter", &P);
      check_positive_finite(function, beta, "Inverse scale parameter", &P);
      check_consistent_sizes(function,
                             n, alpha, beta,
                             "Failures variable",
                             "Shape parameter",
                             "Inverse scale parameter",
                             &P);
          
      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);
          
      // Compute vectorized CDF and gradient
      using stan::math::value_of;
      using stan::math::inc_beta;
      using stan::math::digamma;
      using stan::math::lbeta;
      using std::exp;
      using std::pow;

      agrad::OperandsAndPartials<T_shape, T_inv_scale> 
        operands_and_partials(alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0) 
          return operands_and_partials.to_var(0.0,alpha,beta);
      }
          
      // Cache a few expensive function calls if alpha is a parameter
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape> 
        digammaN_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value, 
                    T_partials_return, T_shape>
        digammaAlpha_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value, 
                    T_partials_return, T_shape> 
        digammaSum_vec(stan::length(alpha));
      if (!is_constant_struct<T_shape>::value) {
              
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const T_partials_return n_dbl = value_of(n_vec[i]);
          const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
                  
          digammaN_vec[i] = digamma(n_dbl + 1);
          digammaAlpha_vec[i] = digamma(alpha_dbl);
          digammaSum_vec[i] = digamma(n_dbl + alpha_dbl + 1);
        }
      }
          
      for (size_t i = 0; i < size; i++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) 
            == std::numeric_limits<int>::max())
          return operands_and_partials.to_var(1.0,alpha,beta);
              
        const T_partials_return n_dbl = value_of(n_vec[i]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
        const T_partials_return beta_dbl = value_of(beta_vec[i]);
              
        const T_partials_return p_dbl = beta_dbl / (1.0 + beta_dbl);
        const T_partials_return d_dbl = 1.0 / ( (1.0 + beta_dbl) 
                                                * (1.0 + beta_dbl) );
              
        const T_partials_return Pi = inc_beta(alpha_dbl, n_dbl + 1.0, p_dbl);
              
        const T_partials_return beta_func = exp(lbeta(n_dbl + 1, alpha_dbl));


        P *= Pi;
              
        if (!is_constant_struct<T_shape>::value) {
                  
          T_partials_return g1 = 0;
          T_partials_return g2 = 0;

          stan::math::grad_reg_inc_beta(g1, g2, alpha_dbl, 
                                        n_dbl + 1, p_dbl, 
                                        digammaAlpha_vec[i], 
                                        digammaN_vec[i], 
                                        digammaSum_vec[i], 
                                        beta_func);
                  
          operands_and_partials.d_x1[i] 
            += g1 / Pi;
        }
              
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]  += d_dbl * pow(1-p_dbl,n_dbl) 
            * pow(p_dbl,alpha_dbl-1) / beta_func / Pi;
              
      }
          
      if (!is_constant_struct<T_shape>::value)
        for(size_t i = 0; i < stan::length(alpha); ++i) 
          operands_and_partials.d_x1[i] *= P;
          
      if (!is_constant_struct<T_inv_scale>::value)
        for(size_t i = 0; i < stan::length(beta); ++i)
          operands_and_partials.d_x2[i] *= P;
          
      return operands_and_partials.to_var(P,alpha,beta);
          
    }

    template <typename T_n, typename T_shape, 
              typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_cdf_log(const T_n& n, const T_shape& alpha, 
                         const T_inv_scale& beta) {
      static const char* function = "stan::prob::neg_binomial_cdf_log(%1%)";
      typedef typename stan::partials_return_type<T_n,T_shape,
                                                  T_inv_scale>::type 
        T_partials_return;

      using stan::math::check_positive_finite;      
      using stan::math::check_nonnegative;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero arugment lengths
      if (!(stan::length(n) && stan::length(alpha) && stan::length(beta)))
        return 0.0;
          
      T_partials_return P(0.0);
          
      // Validate arguments
      check_positive_finite(function, alpha, "Shape parameter", &P);
      check_positive_finite(function, beta, "Inverse scale parameter", &P);
      check_consistent_sizes(function,
                             n, alpha, beta,
                             "Failures variable",
                             "Shape parameter",
                             "Inverse scale parameter",
                             &P);
          
      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);
          
      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      using stan::math::inc_beta;
      using stan::math::digamma;
      using stan::math::lbeta;
      using std::exp;
      using std::pow;

          
      agrad::OperandsAndPartials<T_shape, T_inv_scale> 
        operands_and_partials(alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0) 
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              alpha,beta);
      }
          
      // Cache a few expensive function calls if alpha is a parameter
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape>
        digammaN_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value, 
                    T_partials_return, T_shape> 
        digammaAlpha_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value, 
                    T_partials_return, T_shape> 
        digammaSum_vec(stan::length(alpha));
          
      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const T_partials_return n_dbl = value_of(n_vec[i]);
          const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
                  
          digammaN_vec[i] = digamma(n_dbl + 1);
          digammaAlpha_vec[i] = digamma(alpha_dbl);
          digammaSum_vec[i] = digamma(n_dbl + alpha_dbl + 1);
        }
      }
          
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) == std::numeric_limits<int>::max())
          return operands_and_partials.to_var(0.0,alpha,beta);
              
        const T_partials_return n_dbl = value_of(n_vec[i]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
        const T_partials_return beta_dbl = value_of(beta_vec[i]);
        const T_partials_return p_dbl = beta_dbl / (1.0 + beta_dbl);
        const T_partials_return d_dbl = 1.0 / ( (1.0 + beta_dbl) 
                                                * (1.0 + beta_dbl) );
        const T_partials_return Pi = inc_beta(alpha_dbl, n_dbl + 1.0, p_dbl);
        const T_partials_return beta_func = exp(lbeta(n_dbl + 1, alpha_dbl));


        P += log(Pi);
              
        if (!is_constant_struct<T_shape>::value) {
          T_partials_return g1 = 0;
          T_partials_return g2 = 0;

          stan::math::grad_reg_inc_beta(g1, g2, alpha_dbl, 
                                        n_dbl + 1, p_dbl, 
                                        digammaAlpha_vec[i], 
                                        digammaN_vec[i], 
                                        digammaSum_vec[i], 
                                        beta_func);
          operands_and_partials.d_x1[i] += g1 / Pi;
        }
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]  += d_dbl * pow(1-p_dbl,n_dbl) 
            * pow(p_dbl,alpha_dbl-1) / beta_func / Pi;
      }
          
      return operands_and_partials.to_var(P,alpha,beta);
    }

    template <typename T_n, typename T_shape, 
              typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_ccdf_log(const T_n& n, const T_shape& alpha, 
                          const T_inv_scale& beta) {
      static const char* function = "stan::prob::neg_binomial_ccdf_log(%1%)";
      typedef typename stan::partials_return_type<T_n,T_shape,
                                                  T_inv_scale>::type
        T_partials_return;

      using stan::math::check_positive_finite;      
      using stan::math::check_nonnegative;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero arugment lengths
      if (!(stan::length(n) && stan::length(alpha) && stan::length(beta)))
        return 0.0;
          
      T_partials_return P(0.0);
          
      // Validate arguments
      check_positive_finite(function, alpha, "Shape parameter", &P);
      check_positive_finite(function, beta, "Inverse scale parameter", &P);
      check_consistent_sizes(function,
                             n, alpha, beta,
                             "Failures variable",
                             "Shape parameter",
                             "Inverse scale parameter",
                             &P);
          
      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);
          
      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      using stan::math::inc_beta;
      using stan::math::digamma;
      using stan::math::lbeta;
      using std::exp;
      using std::pow;
          
      agrad::OperandsAndPartials<T_shape, T_inv_scale> 
        operands_and_partials(alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0) 
          return operands_and_partials.to_var(0.0,alpha,beta);
      }
          
      // Cache a few expensive function calls if alpha is a parameter
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape> 
        digammaN_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value, 
                    T_partials_return, T_shape> 
        digammaAlpha_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value, 
                    T_partials_return, T_shape> 
        digammaSum_vec(stan::length(alpha));
          
      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const T_partials_return n_dbl = value_of(n_vec[i]);
          const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
                  
          digammaN_vec[i] = digamma(n_dbl + 1);
          digammaAlpha_vec[i] = digamma(alpha_dbl);
          digammaSum_vec[i] = digamma(n_dbl + alpha_dbl + 1);
        }
      }
          
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) == std::numeric_limits<int>::max())
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              alpha,beta);
              
        const T_partials_return n_dbl = value_of(n_vec[i]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
        const T_partials_return beta_dbl = value_of(beta_vec[i]);
        const T_partials_return p_dbl = beta_dbl / (1.0 + beta_dbl);
        const T_partials_return d_dbl = 1.0 / ( (1.0 + beta_dbl) 
                                                * (1.0 + beta_dbl) );
        const T_partials_return Pi = 1.0 - inc_beta(alpha_dbl, n_dbl + 1.0, 
                                                    p_dbl);
        const T_partials_return beta_func = exp(lbeta(n_dbl + 1, alpha_dbl));

        P += log(Pi);
              
        if (!is_constant_struct<T_shape>::value) {
          T_partials_return g1 = 0;
          T_partials_return g2 = 0;

          stan::math::grad_reg_inc_beta(g1, g2, alpha_dbl, 
                                        n_dbl + 1, p_dbl, 
                                        digammaAlpha_vec[i], 
                                        digammaN_vec[i], 
                                        digammaSum_vec[i], 
                                        beta_func);
          operands_and_partials.d_x1[i] -= g1 / Pi;
        }
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i] -= d_dbl * pow(1-p_dbl,n_dbl) 
            * pow(p_dbl,alpha_dbl-1) / beta_func / Pi;
      }
          
      return operands_and_partials.to_var(P,alpha,beta);
    }
      
    template <class RNG>
    inline int
    neg_binomial_rng(const double alpha,
                     const double beta,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;

      static const char* function = "stan::prob::neg_binomial_rng(%1%)";

      using stan::math::check_positive_finite;      

      check_positive_finite(function, alpha, "Shape parameter", (double*)0);
      check_positive_finite(function, beta, "Inverse scale parameter",
                            (double*)0);

      return stan::prob::poisson_rng(stan::prob::gamma_rng(alpha, beta,
                                                           rng),rng);
    }
  }
}
#endif
