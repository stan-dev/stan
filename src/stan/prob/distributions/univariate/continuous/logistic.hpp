#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__LOGISTIC_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__LOGISTIC_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/functions/constants.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/functions/log1p.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    // Logistic(y|mu,sigma)    [sigma > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const char* function("stan::prob::logistic_log");
      typedef typename stan::partials_return_type<T_y,T_loc,T_scale>::type
        T_partials_return;
      
      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);
        
      // validate args (here done over var, which should be OK)      
      check_finite(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,                             
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;


      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      VectorBuilder<true, T_partials_return, T_scale> inv_sigma(length(sigma));
      VectorBuilder<include_summand<propto,T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
        if (include_summand<propto,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
      }

      VectorBuilder<!is_constant_struct<T_loc>::value, 
                    T_partials_return, T_loc, T_scale>
        exp_mu_div_sigma(max_size(mu,sigma));
      VectorBuilder<!is_constant_struct<T_loc>::value,
                    T_partials_return, T_y, T_scale>
        exp_y_div_sigma(max_size(y,sigma));
      if (!is_constant_struct<T_loc>::value) {
        for (size_t n = 0; n < max_size(mu,sigma); n++) 
          exp_mu_div_sigma[n] = exp(value_of(mu_vec[n]) 
                                    / value_of(sigma_vec[n]));
        for (size_t n = 0; n < max_size(y,sigma); n++) 
          exp_y_div_sigma[n] = exp(value_of(y_vec[n])
                                   / value_of(sigma_vec[n]));
      }

      using stan::math::log1p;
      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
  
        const T_partials_return y_minus_mu = y_dbl - mu_dbl;
        const T_partials_return y_minus_mu_div_sigma = y_minus_mu 
          * inv_sigma[n];
        T_partials_return exp_m_y_minus_mu_div_sigma(0);
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          exp_m_y_minus_mu_div_sigma = exp(-y_minus_mu_div_sigma);
        T_partials_return inv_1p_exp_y_minus_mu_div_sigma(0);
        if (contains_nonconstant_struct<T_y,T_scale>::value)
          inv_1p_exp_y_minus_mu_div_sigma = 1 / (1 + exp(y_minus_mu_div_sigma));

        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp -= y_minus_mu_div_sigma;
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp -= 2.0 * log1p(exp_m_y_minus_mu_div_sigma);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += (2 * inv_1p_exp_y_minus_mu_div_sigma - 1) * inv_sigma[n];
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] +=
            (1 - 2 * exp_mu_div_sigma[n] / (exp_mu_div_sigma[n] 
                                            + exp_y_div_sigma[n]))
            * inv_sigma[n];
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += 
            ((1 - 2 * inv_1p_exp_y_minus_mu_div_sigma)
             *y_minus_mu*inv_sigma[n] - 1) * inv_sigma[n];
      }
      return operands_and_partials.to_var(logp,y,mu,sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return logistic_log<false>(y,mu,sigma);
    }

    // Logistic(y|mu,sigma) [sigma > 0]
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    logistic_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      typedef typename stan::partials_return_type<T_y,T_loc,T_scale>::type 
        T_partials_return;

      // Size checks
      if ( !( stan::length(y) && stan::length(mu) 
              && stan::length(sigma) ) ) 
        return 1.0;
          
      // Error checks
      static const char* function("stan::prob::logistic_cdf");
          
      using stan::math::check_not_nan;
      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using boost::math::tools::promote_args;
          
      T_partials_return P(1.0);
          
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Location parameter", mu, 
                             "Scale parameter", sigma);
          
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
          return operands_and_partials.to_var(0.0,y,mu,sigma);
      }
          
      // Compute vectorized CDF and its gradients
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }
              
        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return sigma_inv_vec = 1.0 / value_of(sigma_vec[n]);
              
        // Compute
        const T_partials_return Pn = 1.0 / ( 1.0 + exp( - (y_dbl - mu_dbl) 
                                                        * sigma_inv_vec ) );
                    
        P *= Pn;
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] 
            += - exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += - (y_dbl - mu_dbl) * sigma_inv_vec 
            * exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
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
          
      return operands_and_partials.to_var(P,y,mu,sigma);
          
    }
      
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    logistic_cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      typedef typename stan::partials_return_type<T_y,T_loc,T_scale>::type 
        T_partials_return;
    
      // Size checks
      if ( !( stan::length(y) && stan::length(mu) && stan::length(sigma) ) )
        return 0.0;
          
      // Error checks
      static const char* function("stan::prob::logistic_cdf_log");
          
      using stan::math::check_not_nan;
      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using boost::math::tools::promote_args;
          
      T_partials_return P(0.0);
          
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Location parameter", mu, 
                             "Scale parameter", sigma);
          
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
          return operands_and_partials.to_var(-std::numeric_limits<double>::infinity(),y,mu,sigma);
      }
          
      // Compute vectorized cdf_log and its gradients
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }
              
        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return sigma_inv_vec = 1.0 / value_of(sigma_vec[n]);
              
        // Compute
        const T_partials_return Pn = 1.0 / ( 1.0 + exp( - (y_dbl - mu_dbl) 
                                                        * sigma_inv_vec ) );
        P += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] 
            += - exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n]  += - (y_dbl - mu_dbl) * sigma_inv_vec 
            * exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
      }
          
      return operands_and_partials.to_var(P,y,mu,sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    logistic_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      typedef typename stan::partials_return_type<T_y,T_loc,T_scale>::type 
        T_partials_return;
  
      // Size checks
      if ( !( stan::length(y) && stan::length(mu) && stan::length(sigma) ) ) 
        return 0.0;
          
      // Error checks
      static const char* function("stan::prob::logistic_cdf_log");
          
      using stan::math::check_not_nan;
      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using boost::math::tools::promote_args;
          
      T_partials_return P(0.0);
          
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu, 
                             "Scale parameter", sigma);
          
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
          return operands_and_partials.to_var(0.0,y,mu,sigma);
      }
          
      // Compute vectorized cdf_log and its gradients
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              y,mu,sigma);
        }
              
        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return sigma_inv_vec = 1.0 / value_of(sigma_vec[n]);
              
        // Compute
        const T_partials_return Pn = 1.0 - 1.0 / (1.0 + exp(-(y_dbl - mu_dbl) 
                                                            * sigma_inv_vec));
        P += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            -= exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] 
            -= - exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= - (y_dbl - mu_dbl) * sigma_inv_vec 
            * exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
      }
          
      return operands_and_partials.to_var(P,y,mu,sigma);
    }

    template <class RNG>
    inline double
    logistic_rng(const double mu,
                 const double sigma,
                 RNG& rng) {
      using boost::variate_generator;
      using boost::random::exponential_distribution;

      static const char* function("stan::prob::logistic_rng");
      
      using stan::math::check_positive_finite;
      using stan::math::check_finite;

      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      variate_generator<RNG&, exponential_distribution<> >
        exp_rng(rng, exponential_distribution<>(1));
      return mu - sigma * std::log(exp_rng() / exp_rng());
    }
  }
}
#endif
