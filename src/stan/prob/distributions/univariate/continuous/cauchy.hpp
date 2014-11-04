#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CAUCHY_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CAUCHY_HPP

#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/log1p.hpp>
#include <stan/math/functions/square.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/functions/log1p.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the Cauchy density for the specified scalar(s) given the specified
     * location parameter(s) and scale parameter(s). y, mu, or sigma can each either be scalar or std::vector.
     * Any vector inputs must be the same length.
     *
     * <p> The result log probability is defined to be the sum of
     * the log probabilities for each observation/mu/sigma triple.
     *
     * @param y (Sequence of) scalar(s).
     * @param mu (Sequence of) location(s).
     * @param sigma (Sequence of) scale(s).
     * @return The log of the product of densities.
     * @tparam T_y Type of scalar outcome.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     * @error_policy
     *    @li sigma must be positive.
     */
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::cauchy_log");

      using stan::is_constant_struct;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return 0.0;
      
      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;

      using stan::math::log1p;
      using stan::math::square;
      
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      DoubleVectorView<true, is_vector<T_scale>::value> inv_sigma(length(sigma));
      DoubleVectorView<true, is_vector<T_scale>::value> sigma_squared(length(sigma));
      DoubleVectorView<include_summand<propto,T_scale>::value,
                       is_vector<T_scale>::value> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        const double sigma_dbl = value_of(sigma_vec[i]);
        inv_sigma[i] = 1.0 / sigma_dbl;
        sigma_squared[i] = sigma_dbl * sigma_dbl;
        if (include_summand<propto,T_scale>::value) {
          log_sigma[i] = log(sigma_dbl);
        }
      }

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> operands_and_partials(y, mu, sigma);

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
  
        // reusable subexpression values
        const double y_minus_mu
          = y_dbl - mu_dbl;
        const double y_minus_mu_squared
          = y_minus_mu * y_minus_mu;
        const double y_minus_mu_over_sigma 
          = y_minus_mu * inv_sigma[n];
        const double y_minus_mu_over_sigma_squared 
          = y_minus_mu_over_sigma * y_minus_mu_over_sigma;

        // log probability
        if (include_summand<propto>::value)
          logp += NEG_LOG_PI;
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp -= log1p(y_minus_mu_over_sigma_squared);
  
        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= 2 * y_minus_mu / (sigma_squared[n] + y_minus_mu_squared);
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += 2 * y_minus_mu / (sigma_squared[n] + y_minus_mu_squared);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += (y_minus_mu_squared - sigma_squared[n]) * inv_sigma[n] / (sigma_squared[n] + y_minus_mu_squared);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return cauchy_log<false>(y,mu,sigma);
    }


    /** 
     * Calculates the cauchy cumulative distribution function for
     * the given variate, location, and scale.
     *
     * \f$\frac{1}{\pi}\arctan\left(\frac{y-\mu}{\sigma}\right) + \frac{1}{2}\f$ 
     *
     * Errors are configured by policy.  All variables must be finite
     * and the scale must be strictly greater than zero.
     *
     * @param y A scalar variate.
     * @param mu The location parameter.
     * @param sigma The scale parameter.
     * 
     * @return 
     */
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma) {
        
      // Size checks
      if ( !( stan::length(y) && stan::length(mu) 
              && stan::length(sigma) ) ) 
        return 1.0;
        
      static const std::string function("stan::prob::cauchy_cdf");
      
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      double P(1.0);
        
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Location parameter", mu, 
                             "Scale Parameter", sigma);
        
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
        
      agrad::OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);
        
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == -std::numeric_limits<double>::infinity()) 
          return operands_and_partials.to_var(0.0);
      }
        
      // Compute CDF and its gradients
      using std::atan;
      using stan::math::pi;

      // Compute vectorized CDF and gradient
      for (size_t n = 0; n < N; n++) {
            
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }
            
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_inv_dbl = 1.0 / value_of(sigma_vec[n]);
            
        const double z = (y_dbl - mu_dbl) * sigma_inv_dbl;
          
        // Compute
        const double Pn = atan(z) / pi() + 0.5;
            
        P *= Pn;
            
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += sigma_inv_dbl / (pi() * (1.0 + z * z) * Pn);
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] 
            += - sigma_inv_dbl / (pi() * (1.0 + z * z) * Pn);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += - z * sigma_inv_dbl / (pi() * (1.0 + z * z) * Pn);
            
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
        for(size_t n = 0; n < stan::length(sigma); ++n) 
          operands_and_partials.d_x3[n] *= P;
      }
        
      return operands_and_partials.to_var(P);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
        
      // Size checks
      if ( !( stan::length(y) && stan::length(mu) 
              && stan::length(sigma) ) ) 
        return 0.0;
        
      static const std::string function("stan::prob::cauchy_cdf");
      
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      double cdf_log(0.0);
        
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Location parameter", mu, 
                             "Scale Parameter", sigma);
        
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
        
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);
        
      // Compute CDFLog and its gradients
      using std::atan;
      using stan::math::pi;

      // Compute vectorized CDF and gradient
      for (size_t n = 0; n < N; n++) {
            
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_inv_dbl = 1.0 / value_of(sigma_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
            
        const double z = (y_dbl - mu_dbl) * sigma_inv_dbl;
          
        // Compute
        const double Pn = atan(z) / pi() + 0.5;
        cdf_log += log(Pn);
            
        const double rep_deriv = 1.0 / (pi() * Pn 
                                        * (z * z * sigma_dbl + sigma_dbl));
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * z;
      }
      return operands_and_partials.to_var(cdf_log);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    cauchy_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
        
      // Size checks
      if ( !( stan::length(y) && stan::length(mu) 
              && stan::length(sigma) ) ) 
        return 0.0;
        
      static const std::string function("stan::prob::cauchy_cdf");
      
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      double ccdf_log(0.0);
        
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Location parameter", mu, 
                             "Scale Parameter", sigma);
        
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
        
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);
        
      // Compute CDFLog and its gradients
      using std::atan;
      using stan::math::pi;

      // Compute vectorized CDF and gradient
      for (size_t n = 0; n < N; n++) {
            
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_inv_dbl = 1.0 / value_of(sigma_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double z = (y_dbl - mu_dbl) * sigma_inv_dbl;
          
        // Compute
        const double Pn = 0.5 - atan(z) / pi();
        ccdf_log += log(Pn);
            
        const double rep_deriv = 1.0 / (Pn * pi() 
                                        * (z * z * sigma_dbl + sigma_dbl));
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += rep_deriv * z;
      }
      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    cauchy_rng(const double mu,
               const double sigma,
               RNG& rng) {
      using boost::variate_generator;
      using boost::random::cauchy_distribution;

      static const std::string function("stan::prob::cauchy_rng");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      variate_generator<RNG&, cauchy_distribution<> >
        cauchy_rng(rng, cauchy_distribution<>(mu, sigma));
      return cauchy_rng();
    }
  }
}
#endif
