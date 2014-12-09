#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__GAMMA_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__GAMMA_HPP

#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_greater_or_equal.hpp>
#include <stan/error_handling/scalar/check_less_or_equal.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/internal_math.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of a gamma density for y with the specified
     * shape and inverse scale parameters.
     * Shape and inverse scale parameters must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Gamma}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\beta^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} \exp^{- \beta y} \right) \\
     &=& \alpha \log(\beta) - \log(\Gamma(\alpha)) + (\alpha - 1) \log(y) - \beta y\\
     & & \mathrm{where} \; y > 0
     \f}
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <bool propto,
              typename T_y, typename T_shape, typename T_inv_scale>
    typename return_type<T_y,T_shape,T_inv_scale>::type
    gamma_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      static const std::string function("stan::prob::gamma_log");

      using stan::is_constant_struct;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
      
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Inverse scale parameter", beta); 
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Shape parameter", alpha,
                             "Inverse scale parameter", beta);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_shape,T_inv_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);

      for (size_t n = 0; n < length(y); n++) {
        const double y_dbl = value_of(y_vec[n]);
        if (y_dbl < 0)
          return LOG_ZERO;
      }

      size_t N = max_size(y, alpha, beta);
      agrad::OperandsAndPartials<T_y, T_shape, T_inv_scale> operands_and_partials(y, alpha, beta);

      using boost::math::lgamma;
      using stan::math::multiply_log;
      using boost::math::digamma;
      
      DoubleVectorView<include_summand<propto,T_y,T_shape>::value,is_vector<T_y>::value>
        log_y(length(y));
      if (include_summand<propto,T_y,T_shape>::value)
        for(size_t n = 0; n < length(y); n++) {
          if (value_of(y_vec[n]) > 0)
            log_y[n] = log(value_of(y_vec[n]));
        }

      DoubleVectorView<include_summand<propto,T_shape>::value,is_vector<T_shape>::value>
        lgamma_alpha(length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,is_vector<T_shape>::value>
        digamma_alpha(length(alpha));
      for (size_t n = 0; n < length(alpha); n++) {
        if (include_summand<propto,T_shape>::value)
          lgamma_alpha[n] = lgamma(value_of(alpha_vec[n]));
        if (!is_constant_struct<T_shape>::value)
          digamma_alpha[n] = digamma(value_of(alpha_vec[n]));
      }

      DoubleVectorView<include_summand<propto,T_shape,T_inv_scale>::value,is_vector<T_inv_scale>::value>
        log_beta(length(beta));
      if (include_summand<propto,T_shape,T_inv_scale>::value)
        for (size_t n = 0; n < length(beta); n++)
          log_beta[n] = log(value_of(beta_vec[n]));
      
      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
  
        if (include_summand<propto,T_shape>::value)
          logp -= lgamma_alpha[n];
        if (include_summand<propto,T_shape,T_inv_scale>::value)
          logp += alpha_dbl * log_beta[n];
        if (include_summand<propto,T_y,T_shape>::value)
          logp += (alpha_dbl-1.0) * log_y[n];
        if (include_summand<propto,T_y,T_inv_scale>::value)
          logp -= beta_dbl * y_dbl;
  
        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += (alpha_dbl-1)/y_dbl - beta_dbl;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n] += -digamma_alpha[n] + log_beta[n] + log_y[n];
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x3[n] += alpha_dbl / beta_dbl - y_dbl;
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_shape, typename T_inv_scale>
    inline
    typename return_type<T_y,T_shape,T_inv_scale>::type
    gamma_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      return gamma_log<false>(y,alpha,beta);
    }


    /**
     * The cumulative density function for a gamma distribution for y
     * with the specified shape and inverse scale parameters.
     *
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */    
    template <typename T_y, typename T_shape, typename T_inv_scale>
    typename return_type<T_y,T_shape,T_inv_scale>::type
    gamma_cdf(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      // Size checks
      if (!(stan::length(y) && stan::length(alpha) && stan::length(beta))) 
        return 1.0;
          
      // Error checks
      static const std::string function("stan::prob::gamma_cdf");
          
      using stan::error_handling::check_positive_finite;      
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_less_or_equal;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using boost::math::tools::promote_args;
          
      double P(1.0);
          
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", beta);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y); 
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Shape parameter", alpha, 
                             "Scale Parameter", beta);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);
          
      agrad::OperandsAndPartials<T_y, T_shape, T_inv_scale> 
        operands_and_partials(y, alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
          
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(0.0);
      }
          
      // Compute CDF and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_p;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_shape>::value,
                       is_vector<T_shape>::value> 
        gamma_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
                       is_vector<T_shape>::value> 
        digamma_vec(stan::length(alpha));
          
      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const double alpha_dbl = value_of(alpha_vec[i]);
          gamma_vec[i] = tgamma(alpha_dbl);
          digamma_vec[i] = digamma(alpha_dbl);
        }
      }
          
      // Compute vectorized CDF and gradient
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity())
          continue;
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
              
        // Compute
        const double Pn = gamma_p(alpha_dbl, beta_dbl * y_dbl);
              
        P *= Pn;
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += beta_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) 
            / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n] 
            -= stan::math::gradRegIncGamma(alpha_dbl, beta_dbl
                                           * y_dbl, gamma_vec[n],
                                           digamma_vec[n]) / Pn;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x3[n] 
            += y_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) / Pn;
      }
          
      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= P;
      if (!is_constant_struct<T_shape>::value)
        for (size_t n = 0; n < stan::length(alpha); ++n) 
          operands_and_partials.d_x2[n] *= P;
      if (!is_constant_struct<T_inv_scale>::value)
        for (size_t n = 0; n < stan::length(beta); ++n) 
          operands_and_partials.d_x3[n] *= P;
          
      return operands_and_partials.to_var(P);
    }

    template <typename T_y, typename T_shape, typename T_inv_scale>
    typename return_type<T_y,T_shape,T_inv_scale>::type
    gamma_cdf_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      // Size checks
      if (!(stan::length(y) && stan::length(alpha) && stan::length(beta))) 
        return 0.0;
          
      // Error checks
      static const std::string function("stan::prob::gamma_cdf_log");
          
      using stan::error_handling::check_positive_finite;      
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_less_or_equal;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using boost::math::tools::promote_args;
          
      double P(0.0);
          
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", beta);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y); 
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Shape parameter", alpha, 
                             "Scale Parameter", beta);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);
          
      agrad::OperandsAndPartials<T_y, T_shape, T_inv_scale> 
        operands_and_partials(y, alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
          
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
          
      // Compute cdf_log and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_p;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_shape>::value,
                       is_vector<T_shape>::value> 
        gamma_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
                       is_vector<T_shape>::value> 
        digamma_vec(stan::length(alpha));
          
      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const double alpha_dbl = value_of(alpha_vec[i]);
          gamma_vec[i] = tgamma(alpha_dbl);
          digamma_vec[i] = digamma(alpha_dbl);
        }
      }
          
      // Compute vectorized cdf_log and gradient
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity())
          return operands_and_partials.to_var(0.0);
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
              
        // Compute
        const double Pn = gamma_p(alpha_dbl, beta_dbl * y_dbl);
              
        P += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += beta_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) 
            / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n] 
            -= stan::math::gradRegIncGamma(alpha_dbl, beta_dbl
                                           * y_dbl, gamma_vec[n],
                                           digamma_vec[n]) / Pn;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x3[n] 
            += y_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) / Pn;
      }
          
      return operands_and_partials.to_var(P);
    }

 template <typename T_y, typename T_shape, typename T_inv_scale>
    typename return_type<T_y,T_shape,T_inv_scale>::type
    gamma_ccdf_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      // Size checks
      if (!(stan::length(y) && stan::length(alpha) && stan::length(beta))) 
        return 0.0;
          
      // Error checks
      static const std::string function("stan::prob::gamma_ccdf_log");
          
      using stan::error_handling::check_positive_finite;      
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_less_or_equal;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using boost::math::tools::promote_args;
          
      double P(0.0);
          
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", beta);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y); 
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Shape parameter", alpha, 
                             "Scale Parameter", beta);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);
          
      agrad::OperandsAndPartials<T_y, T_shape, T_inv_scale> 
        operands_and_partials(y, alpha, beta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
          
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(0.0);
      }
          
      // Compute ccdf_log and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_p;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_shape>::value,
                       is_vector<T_shape>::value> 
        gamma_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
                       is_vector<T_shape>::value> 
        digamma_vec(stan::length(alpha));
          
      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const double alpha_dbl = value_of(alpha_vec[i]);
          gamma_vec[i] = tgamma(alpha_dbl);
          digamma_vec[i] = digamma(alpha_dbl);
        }
      }
          
      // Compute vectorized ccdf_log and gradient
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity())
          return operands_and_partials.to_var(stan::math::negative_infinity());
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
              
        // Compute
        const double Pn = 1.0 - gamma_p(alpha_dbl, beta_dbl * y_dbl);
              
        P += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            -= beta_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) 
            / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n] 
            += stan::math::gradRegIncGamma(alpha_dbl, beta_dbl
                                           * y_dbl, gamma_vec[n],
                                           digamma_vec[n]) / Pn;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x3[n] 
            -= y_dbl * gamma_p_derivative(alpha_dbl, beta_dbl * y_dbl) / Pn;
      }
          
      return operands_and_partials.to_var(P);
    }

    template <class RNG>
    inline double
    gamma_rng(const double alpha,
              const double beta,
              RNG& rng) {
      using boost::variate_generator;
      using boost::gamma_distribution;

      static const std::string function("stan::prob::gamma_rng");

      using stan::error_handling::check_positive_finite;
      
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Inverse scale parameter", beta);

      /*
        the boost gamma distribution is defined by
        shape and scale, whereas the stan one is defined
        by shape and rate
      */
      variate_generator<RNG&, gamma_distribution<> >
        gamma_rng(rng, gamma_distribution<>(alpha, 1.0 / beta));
      return gamma_rng();
    }

  }
}

#endif
