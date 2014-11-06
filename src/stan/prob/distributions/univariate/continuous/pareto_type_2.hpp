#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__PARETO_TYPE_2_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__PARETO_TYPE_2_HPP

#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_greater_or_equal.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/functions/log1m.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>
#include <stan/prob/traits.hpp>


namespace stan {
  namespace prob {

    // pareto_type_2(y|lambda,alpha)  [y >= 0;  lambda > 0;  alpha > 0]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y,T_loc,T_scale,T_shape>::type
    pareto_type_2_log(const T_y& y, const T_loc& mu, const T_scale& lambda, 
              const T_shape& alpha) {
      static const std::string function("stan::prob::pareto_type_2_log");
      
      using std::log;
      using stan::math::value_of;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(lambda) 
            && stan::length(alpha)))
        return 0.0;
      
      // set up return value accumulator
      double logp(0.0);
      
      // validate args (here done over var, which should be OK)
      check_greater_or_equal(function, "Random variable", y, mu);
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", lambda);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Scale parameter", lambda,
                             "Shape parameter", alpha);


      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_scale,T_shape>::value)
        return 0.0;
      
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> lambda_vec(lambda);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, lambda, alpha);

      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y,T_loc,T_scale,T_shape> 
        operands_and_partials(y, mu, lambda, alpha);
      
      DoubleVectorView<include_summand<propto,T_y,T_loc,T_scale,T_shape>::value,
                       is_vector<T_y>::value> log1p_scaled_diff(N);
      if (include_summand<propto,T_y,T_loc,T_scale,T_shape>::value)
        for (size_t n = 0; n < N; n++)
          log1p_scaled_diff[n] = log1p((value_of(y_vec[n]) 
                                            - value_of(mu_vec[n]))
                                       / value_of(lambda_vec[n]));

      DoubleVectorView<include_summand<propto,T_scale>::value,
                       is_vector<T_scale>::value> log_lambda(length(lambda));
      if (include_summand<propto,T_scale>::value)
        for (size_t n = 0; n < length(lambda); n++)
          log_lambda[n] = log(value_of(lambda_vec[n]));

      DoubleVectorView<include_summand<propto,T_shape>::value,
                       is_vector<T_shape>::value> log_alpha(length(alpha));
      if (include_summand<propto,T_shape>::value)
        for (size_t n = 0; n < length(alpha); n++)
          log_alpha[n] = log(value_of(alpha_vec[n]));

      DoubleVectorView<!is_constant_struct<T_shape>::value,
                       is_vector<T_shape>::value> inv_alpha(length(alpha));
      if (!is_constant_struct<T_shape>::value)
        for (size_t n = 0; n < length(alpha); n++)
          inv_alpha[n] = 1 / value_of(alpha_vec[n]);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double sum_dbl = lambda_dbl + y_dbl + mu_dbl;
        const double inv_sum = 1.0 / sum_dbl;
        const double alpha_div_sum = alpha_dbl / sum_dbl;
        const double deriv_1_2 = inv_sum + alpha_div_sum;

        // // log probability
        if (include_summand<propto,T_shape>::value)
          logp += log_alpha[n];
        if (include_summand<propto,T_scale>::value)
          logp -= log_lambda[n];
        if (include_summand<propto,T_y,T_scale,T_shape>::value)
          logp -= (alpha_dbl + 1.0) * log1p_scaled_diff[n];
  
        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= deriv_1_2;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += deriv_1_2;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= alpha_div_sum * (mu_dbl - y_dbl)
            / lambda_dbl + inv_sum;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] += inv_alpha[n] - log1p_scaled_diff[n];
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline
    typename return_type<T_y,T_loc,T_scale,T_shape>::type
    pareto_type_2_log(const T_y& y, const T_loc& mu, 
              const T_scale& lambda, const T_shape& alpha) {
      return pareto_type_2_log<false>(y,mu,lambda,alpha);
    }
    
    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y, T_loc, T_scale, T_shape>::type
    pareto_type_2_cdf(const T_y& y, const T_loc& mu, 
              const T_scale& lambda, const T_shape& alpha) {
          
      // Check sizes
      // Size checks
      if ( !( stan::length(y) 
              && stan::length(mu)
              && stan::length(lambda) 
              && stan::length(alpha) ) )
        return 1.0;
          
      // Check errors
      static const std::string function("stan::prob::pareto_type_2_cdf");
          
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
          
      double P(1.0);
          
      check_greater_or_equal(function, "Random variable", y, mu);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", lambda);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Scale parameter", lambda, 
                             "Shape parameter", alpha);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> lambda_vec(lambda);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, lambda, alpha);
          
      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_shape> 
        operands_and_partials(y, mu, lambda, alpha);

      DoubleVectorView<true, is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        p1_pow_alpha(N);

      DoubleVectorView<!is_constant_struct<T_y>::value
                       || !is_constant_struct<T_loc>::value
                       || !is_constant_struct<T_scale>::value, 
                       is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        grad_1_2(N);

      DoubleVectorView<!is_constant_struct<T_shape>::value, 
                       is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        grad_3(N);

      for (size_t i = 0; i < N; i++) {
        const double lambda_dbl = value_of(lambda_vec[i]);
        const double alpha_dbl = value_of(alpha_vec[i]);
        const double temp = 1 + (value_of(y_vec[i]) 
                                 - value_of(mu_vec[i])) 
          / lambda_dbl;
        p1_pow_alpha[i] = pow(temp, -alpha_dbl);

        if (!is_constant_struct<T_y>::value 
            || !is_constant_struct<T_loc>::value
            || !is_constant_struct<T_scale>::value)
          grad_1_2[i] = p1_pow_alpha[i] / temp * alpha_dbl / lambda_dbl;

        if (!is_constant_struct<T_shape>::value)
          grad_3[i] = log(temp) * p1_pow_alpha[i];
      }

      // Compute vectorized CDF and its gradients
          
      for (size_t n = 0; n < N; n++) {
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);
              
        const double Pn = 1.0 - p1_pow_alpha[n];

        // Compute
        P *= Pn;
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += grad_1_2[n] / Pn;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= grad_1_2[n] / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += (mu_dbl - y_dbl)
            * grad_1_2[n] / lambda_dbl / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] += grad_3[n] / Pn;
      }
          
      if (!is_constant_struct<T_y>::value) {
        for(size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= P;
      }
      if (!is_constant_struct<T_loc>::value) {
        for(size_t n = 0; n < stan::length(mu); ++n) 
          operands_and_partials.d_x2[n] *= P;
      }
      if (!is_constant_struct<T_scale>::value) {
        for(size_t n = 0; n < stan::length(lambda); ++n) 
          operands_and_partials.d_x3[n] *= P;
      }
      if (!is_constant_struct<T_shape>::value) {
        for(size_t n = 0; n < stan::length(alpha); ++n)
          operands_and_partials.d_x4[n] *= P;
      }
          
      return operands_and_partials.to_var(P);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y, T_loc, T_scale, T_shape>::type
    pareto_type_2_cdf_log(const T_y& y, const T_loc& mu, 
                  const T_scale& lambda, const T_shape& alpha) {
          
      // Check sizes
      // Size checks
      if ( !( stan::length(y) 
              && stan::length(mu)
              && stan::length(lambda) 
              && stan::length(alpha) ) )
        return 0.0;
          
      // Check errors
      static const std::string function("stan::prob::pareto_type_2_cdf_log");
          
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::math::log1m;

      double P(0.0);

      check_greater_or_equal(function, "Random variable", y, mu);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", lambda);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Scale parameter", lambda, 
                             "Shape parameter", alpha);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> lambda_vec(lambda);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, lambda, alpha);
          
      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_shape> 
        operands_and_partials(y, mu, lambda, alpha);

      DoubleVectorView<true, 
                       is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        cdf_log(N);

      DoubleVectorView<true, 
                       is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        inv_p1_pow_alpha_minus_one(N);

      DoubleVectorView<!is_constant_struct<T_shape>::value, 
                       is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        log_1p_y_over_lambda(N);

      for (size_t i = 0; i < N; i++) {
        const double temp = 1.0 + (value_of(y_vec[i]) 
                                   - value_of(mu_vec[i])) 
          / value_of(lambda_vec[i]);
        const double p1_pow_alpha = pow(temp, value_of(alpha_vec[i]));
        cdf_log[i] = log1m(1.0 / p1_pow_alpha);

        inv_p1_pow_alpha_minus_one[i] = 1.0 / (p1_pow_alpha - 1.0);

        if (!is_constant_struct<T_shape>::value)
          log_1p_y_over_lambda[i] = log(temp);
      }

      // Compute vectorized CDF and its gradients
          
      for (size_t n = 0; n < N; n++) {
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);

        const double grad_1_2 =  alpha_dbl 
          * inv_p1_pow_alpha_minus_one[n] / (lambda_dbl - mu_dbl + y_dbl);
              
        // Compute
        P += cdf_log[n];
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += grad_1_2;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= grad_1_2;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += (mu_dbl - y_dbl) * grad_1_2 
            / lambda_dbl;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] += log_1p_y_over_lambda[n] 
            * inv_p1_pow_alpha_minus_one[n];
      }
          
      return operands_and_partials.to_var(P);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y, T_loc, T_scale, T_shape>::type
    pareto_type_2_ccdf_log(const T_y& y, const T_loc& mu,
                   const T_scale& lambda, const T_shape& alpha) {
          
      // Check sizes
      // Size checks
      if ( !( stan::length(y)
              && stan::length(mu)
              && stan::length(lambda) 
              && stan::length(alpha) ) )
        return 0.0;
          
      // Check errors
      static const std::string function("stan::prob::pareto_type_2_ccdf_log");
          
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
          
      double P(0.0);
          
      check_greater_or_equal(function, "Random variable", y, mu);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", lambda);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Scale parameter", lambda, 
                             "Shape parameter", alpha);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> lambda_vec(lambda);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, lambda, alpha);
          
      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_shape> 
        operands_and_partials(y, mu, lambda, alpha);

      DoubleVectorView<true, 
                       is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        ccdf_log(N);

      DoubleVectorView<!is_constant_struct<T_y>::value 
                       || !is_constant_struct<T_loc>::value 
                       || !is_constant_struct<T_scale>::value 
                       || !is_constant_struct<T_shape>::value, 
                       is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        a_over_lambda_plus_y(N);

      DoubleVectorView<!is_constant_struct<T_shape>::value, 
                       is_vector<T_y>::value 
                       || is_vector<T_loc>::value
                       || is_vector<T_scale>::value
                       || is_vector<T_shape>::value>
        log_1p_y_over_lambda(N);

      for (size_t i = 0; i < N; i++) {
        const double y_dbl = value_of(y_vec[i]);
        const double mu_dbl = value_of(mu_vec[i]);
        const double lambda_dbl = value_of(lambda_vec[i]);
        const double alpha_dbl = value_of(alpha_vec[i]);
        const double temp = 1.0 + (y_dbl - mu_dbl) / lambda_dbl;
        const double log_temp = log(temp);

        ccdf_log[i] = -alpha_dbl * log_temp;

        if (!is_constant_struct<T_y>::value 
            || !is_constant_struct<T_loc>::value 
            || !is_constant_struct<T_scale>::value 
            || !is_constant_struct<T_shape>::value)
          a_over_lambda_plus_y[i] = alpha_dbl / (y_dbl - mu_dbl + lambda_dbl);

        if (!is_constant_struct<T_shape>::value)
          log_1p_y_over_lambda[i] = log_temp;
      }

      // Compute vectorized CDF and its gradients
          
      for (size_t n = 0; n < N; n++) {
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);
              
        // Compute
        P += ccdf_log[n];
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= a_over_lambda_plus_y[n];
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += a_over_lambda_plus_y[n];
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += a_over_lambda_plus_y[n] 
            * (y_dbl - mu_dbl) / lambda_dbl;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] -= log_1p_y_over_lambda[n];
      }
          
      return operands_and_partials.to_var(P);
    }

    template <class RNG>
    inline double
    pareto_type_2_rng(const double mu,
                      const double lambda,
                      const double alpha,
                      RNG& rng) {
      static const std::string function("stan::prob::pareto_type_2_rng");
      
      stan::error_handling::check_positive(function, "scale parameter", lambda);

      double uniform_01 = stan::prob::uniform_rng(0.0, 1.0, rng);


      return (std::pow(1.0 - uniform_01, -1.0 / alpha) - 1.0) * lambda + mu;
    }
  }
}
#endif
