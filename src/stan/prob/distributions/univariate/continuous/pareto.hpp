#ifndef STAN__PROB__DISTRIBUTIONS__PARETO_HPP
#define STAN__PROB__DISTRIBUTIONS__PARETO_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_greater_or_equal.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>


namespace stan {
  namespace prob {

    // Pareto(y|y_m,alpha)  [y > y_m;  y_m > 0;  alpha > 0]
    template <bool propto,
              typename T_y, typename T_scale, typename T_shape>
    typename return_type<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
      static const std::string function("stan::prob::pareto_log");
      
      using stan::math::value_of;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(y_min) 
            && stan::length(alpha)))
        return 0.0;
      
      // set up return value accumulator
      double logp(0.0);
      
      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", y_min);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Scale parameter", y_min,
                             "Shape parameter", alpha);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_scale,T_shape>::value)
        return 0.0;
      
      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> y_min_vec(y_min);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, y_min, alpha);

      for (size_t n = 0; n < N; n++) {
        if (y_vec[n] < y_min_vec[n])
          return LOG_ZERO;
      }

      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y,T_scale,T_shape> 
        operands_and_partials(y, y_min, alpha);
      
      DoubleVectorView<include_summand<propto,T_y,T_shape>::value,
                       is_vector<T_y>::value> log_y(length(y));
      if (include_summand<propto,T_y,T_shape>::value)
        for (size_t n = 0; n < length(y); n++)
          log_y[n] = log(value_of(y_vec[n]));

      DoubleVectorView<!is_constant_struct<T_y>::value
                       ||!is_constant_struct<T_shape>::value,
                       is_vector<T_y>::value> inv_y(length(y));
      if (!is_constant_struct<T_y>::value||!is_constant_struct<T_shape>::value)
        for (size_t n = 0; n < length(y); n++)
          inv_y[n] = 1 / value_of(y_vec[n]);

      DoubleVectorView<include_summand<propto,T_scale,T_shape>::value,
                       is_vector<T_scale>::value> 
        log_y_min(length(y_min));
      if (include_summand<propto,T_scale,T_shape>::value)
        for (size_t n = 0; n < length(y_min); n++)
          log_y_min[n] = log(value_of(y_min_vec[n]));

      DoubleVectorView<include_summand<propto,T_shape>::value,
                       is_vector<T_shape>::value> log_alpha(length(alpha));
      if (include_summand<propto,T_shape>::value)
        for (size_t n = 0; n < length(alpha); n++)
          log_alpha[n] = log(value_of(alpha_vec[n]));
      
      using stan::math::multiply_log;

      for (size_t n = 0; n < N; n++) {
        const double alpha_dbl = value_of(alpha_vec[n]);
        // log probability
        if (include_summand<propto,T_shape>::value)
          logp += log_alpha[n];
        if (include_summand<propto,T_scale,T_shape>::value)
          logp += alpha_dbl * log_y_min[n];
        if (include_summand<propto,T_y,T_shape>::value)
          logp -= alpha_dbl * log_y[n] + log_y[n];
  
        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= alpha_dbl * inv_y[n] + inv_y[n];
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] += alpha_dbl / value_of(y_min_vec[n]);
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x3[n] 
            += 1 / alpha_dbl + log_y_min[n] - log_y[n];
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_scale, typename T_shape>
    inline
    typename return_type<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
      return pareto_log<false>(y,y_min,alpha);
    }

    template <typename T_y, typename T_scale, typename T_shape>
    typename return_type<T_y, T_scale, T_shape>::type
    pareto_cdf(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
          
      // Check sizes
      // Size checks
      if ( !( stan::length(y) && stan::length(y_min) && stan::length(alpha) ) )
        return 1.0;
          
      // Check errors
      static const std::string function("stan::prob::pareto_cdf");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
          
      double P(1.0);
          
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", y_min);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Scale parameter", y_min, 
                             "Shape parameter", alpha);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> y_min_vec(y_min);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, y_min, alpha);
          
      agrad::OperandsAndPartials<T_y, T_scale, T_shape> 
        operands_and_partials(y, y_min, alpha);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
          
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) < value_of(y_min_vec[i])) 
          return operands_and_partials.to_var(0.0);
      }
          
      // Compute vectorized CDF and its gradients
          
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }
              
        // Pull out values
        const double log_dbl = log( value_of(y_min_vec[n]) 
                                    / value_of(y_vec[n]) );
        const double y_min_inv_dbl = 1.0 / value_of(y_min_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
              
        // Compute
        const double Pn = 1.0 - exp( alpha_dbl * log_dbl );
                    
        P *= Pn;
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += alpha_dbl * y_min_inv_dbl * exp( (alpha_dbl + 1) * log_dbl )
            / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] 
            += - alpha_dbl * y_min_inv_dbl * exp( alpha_dbl * log_dbl ) / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x3[n] 
            += - exp( alpha_dbl * log_dbl ) * log_dbl / Pn;
      }
          
      if (!is_constant_struct<T_y>::value) {
        for(size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= P;
      }
      if (!is_constant_struct<T_scale>::value) {
        for(size_t n = 0; n < stan::length(y_min); ++n) 
          operands_and_partials.d_x2[n] *= P;
      }
      if (!is_constant_struct<T_shape>::value) {
        for(size_t n = 0; n < stan::length(alpha); ++n)
          operands_and_partials.d_x3[n] *= P;
      }
          
      return operands_and_partials.to_var(P);
    }

    template <typename T_y, typename T_scale, typename T_shape>
    typename return_type<T_y, T_scale, T_shape>::type
    pareto_cdf_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
          
      // Size checks
      if ( !( stan::length(y) && stan::length(y_min) && stan::length(alpha) ) )
        return 0.0;
          
      // Check errors
      static const std::string function("stan::prob::pareto_cdf_log");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
          
      double P(0.0);
          
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", y_min);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Scale parameter", y_min, 
                             "Shape parameter", alpha);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> y_min_vec(y_min);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, y_min, alpha);
          
      agrad::OperandsAndPartials<T_y, T_scale, T_shape> 
        operands_and_partials(y, y_min, alpha);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
          
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) < value_of(y_min_vec[i])) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
          
      // Compute vectorized cdf_log and its gradients
          
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          return operands_and_partials.to_var(0.0);
        }
              
        // Pull out values
        const double log_dbl = log( value_of(y_min_vec[n]) 
                                    / value_of(y_vec[n]) );
        const double y_min_inv_dbl = 1.0 / value_of(y_min_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
              
        // Compute
        const double Pn = 1.0 - exp(alpha_dbl * log_dbl );
                    
        P += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += alpha_dbl * y_min_inv_dbl * exp((alpha_dbl + 1) * log_dbl) / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] 
            -= alpha_dbl * y_min_inv_dbl * exp( alpha_dbl * log_dbl ) / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x3[n] 
            -= exp( alpha_dbl * log_dbl ) * log_dbl / Pn;
      }
          
      return operands_and_partials.to_var(P);
    }

    template <typename T_y, typename T_scale, typename T_shape>
    typename return_type<T_y, T_scale, T_shape>::type
    pareto_ccdf_log(const T_y& y, const T_scale& y_min,
                    const T_shape& alpha) {
          
      // Size checks
      if ( !( stan::length(y) && stan::length(y_min) && stan::length(alpha) ) )
        return 0.0;
          
      // Check errors
      static const std::string function("stan::prob::pareto_ccdf_log");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
          
      double P(0.0);
          
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", y_min);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Scale parameter", y_min, 
                             "Shape parameter", alpha);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> y_min_vec(y_min);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, y_min, alpha);
          
      agrad::OperandsAndPartials<T_y, T_scale, T_shape> 
        operands_and_partials(y, y_min, alpha);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
          
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) < value_of(y_min_vec[i])) 
          return operands_and_partials.to_var(0.0);
      }
          
      // Compute vectorized cdf_log and its gradients
          
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          return operands_and_partials.to_var(stan::math::negative_infinity());
        }
              
        // Pull out values
        const double log_dbl = log( value_of(y_min_vec[n]) 
                                    / value_of(y_vec[n]) );
        const double y_min_inv_dbl = 1.0 / value_of(y_min_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
              
        P += alpha_dbl * log_dbl;
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= alpha_dbl * y_min_inv_dbl 
            * exp(log_dbl);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] += alpha_dbl * y_min_inv_dbl;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x3[n] += log_dbl;
      }
          
      return operands_and_partials.to_var(P);
    }
      
    template <class RNG>
    inline double
    pareto_rng(const double y_min,
               const double alpha,
               RNG& rng) {
      using boost::variate_generator;
      using boost::exponential_distribution;

      static const std::string function("stan::prob::pareto_rng");
      
      using stan::error_handling::check_positive_finite;

      check_positive_finite(function, "Scale parameter", y_min);
      check_positive_finite(function, "Shape parameter", alpha);

      variate_generator<RNG&, exponential_distribution<> >
        exp_rng(rng, exponential_distribution<>(alpha));
      return y_min * std::exp(exp_rng());
    }
  }
}
#endif
