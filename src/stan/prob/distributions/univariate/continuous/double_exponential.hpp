#ifndef __STAN__PROB__DISTRIBUTIONS__DOUBLE_EXPONENTIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__DOUBLE_EXPONENTIAL_HPP__

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {
    template<typename T>
    inline int sign(const T& z) {
      return (z == 0) ? 0 : z < 0 ? -1 : 1;
    }

    // DoubleExponential(y|mu,sigma)  [sigma > 0]
    // FIXME: add documentation
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, 
                           const T_loc& mu, const T_scale& sigma) {
      static const char* function
        = "stan::prob::double_exponential_log(%1%)";
      
      using stan::is_constant_struct;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;
      using std::log;
      using std::fabs;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);
      if(!check_finite(function, y, "Random variable", &logp))
        return logp;
      if(!check_finite(function, mu, "Location parameter", 
                       &logp))
        return logp;
      if(!check_finite(function, sigma, "Scale parameter", 
                       &logp))
        return logp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,
                                   "Random variable","Location parameter","Shape parameter",
                                   &logp)))
        return logp;
      
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      agrad::OperandsAndPartials<T_y,T_loc,T_scale> operands_and_partials(y, mu, sigma);

      DoubleVectorView<include_summand<propto,T_y,T_loc,T_scale>::value,is_vector<T_scale>::value> 
        inv_sigma(length(sigma));
      DoubleVectorView<!is_constant_struct<T_scale>::value,is_vector<T_scale>::value> 
        inv_sigma_squared(length(sigma));
      DoubleVectorView<include_summand<propto,T_scale>::value,is_vector<T_scale>::value> 
        log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        const double sigma_dbl = value_of(sigma_vec[i]);
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          inv_sigma[i] = 1.0 / sigma_dbl;
        if (include_summand<propto,T_scale>::value) 
          log_sigma[i] = log(value_of(sigma_vec[i]));
        if (!is_constant_struct<T_scale>::value) 
          inv_sigma_squared[i] = inv_sigma[i] * inv_sigma[i];
      }


      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
  
        // reusable subexpressions values
        const double y_m_mu = y_dbl - mu_dbl;
        const double fabs_y_m_mu = fabs(y_m_mu);

        // log probability
        if (include_summand<propto>::value)
          logp += NEG_LOG_TWO;
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp -= fabs_y_m_mu * inv_sigma[n];
  
        // gradients
        double sign_y_m_mu_times_inv_sigma(0);
        if (!is_constant_struct<T_y>::value || !is_constant_struct<T_loc>::value)
          sign_y_m_mu_times_inv_sigma = sign(y_m_mu) * inv_sigma[n];
        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n] -= sign_y_m_mu_times_inv_sigma;
        }
        if (!is_constant_struct<T_loc>::value) {
          operands_and_partials.d_x2[n] += sign_y_m_mu_times_inv_sigma;
        }
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += -inv_sigma[n] + fabs_y_m_mu * inv_sigma_squared[n];
      }
      return operands_and_partials.to_var(logp);
    }


    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, const T_loc& mu, 
                           const T_scale& sigma) {
      return double_exponential_log<false>(y,mu,sigma);
    }

    /** 
     * Calculates the double exponential cumulative density function.
     *
     * \f$ f(y|\mu,\sigma) = \begin{cases} \
     \frac{1}{2} \exp\left(\frac{y-\mu}{\sigma}\right), \mbox{if } y < \mu \\ 
     1 - \frac{1}{2} \exp\left(-\frac{y-\mu}{\sigma}\right), \mbox{if } y \ge \mu \
     \end{cases}\f$
     * 
     * @param y A scalar variate.
     * @param mu The location parameter.
     * @param sigma The scale parameter.
     * 
     * @return The cumulative density function.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    double_exponential_cdf(const T_y& y, 
                           const T_loc& mu, const T_scale& sigma) {
      static const char* function
        = "stan::prob::double_exponential_cdf(%1%)";
      
      // Size checks
      if ( !( stan::length(y) && stan::length(mu) 
              && stan::length(sigma) ) ) 
        return 1.0;
        
      using stan::math::value_of;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;

      double cdf(1.0);
      if(!check_not_nan(function, y, "Random variable", &cdf))
        return cdf;
      if(!check_finite(function, mu, "Location parameter", &cdf))
        return cdf;
      if(!check_finite(function, sigma, "Scale parameter", &cdf))
        return cdf;
      if(!check_positive(function, sigma, "Scale parameter", &cdf))
        return cdf;

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      //cdf
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / (sigma_dbl);
        const double exp_scaled_diff = exp(scaled_diff);

        if(y_dbl < mu_dbl) {
          cdf *= exp_scaled_diff * 0.5;
        }
        else
          cdf *= 1.0 - 0.5 / exp_scaled_diff;
      }

      //gradients
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / sigma_dbl;
        const double exp_scaled_diff = exp(scaled_diff);
        const double inv_sigma = 1.0 / sigma_dbl;
        
        if(y_dbl < mu_dbl) {
          if (!is_constant_struct<T_y>::value)
            operands_and_partials.d_x1[n] += inv_sigma * cdf;
          if (!is_constant_struct<T_loc>::value)
            operands_and_partials.d_x2[n] -= inv_sigma * cdf;
          if (!is_constant_struct<T_scale>::value)
            operands_and_partials.d_x3[n] -= scaled_diff * inv_sigma  * cdf;
        }
        else {
          const double rep_deriv = cdf * inv_sigma / (2.0 * exp_scaled_diff
                                                      - 1.0);
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff;
        }
      }
      return operands_and_partials.to_var(cdf);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    double_exponential_cdf_log(const T_y& y, const T_loc& mu, 
                               const T_scale& sigma) {
      static const char* function
        = "stan::prob::double_exponential_cdf_log(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_positive;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      double cdf_log(0.0);

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf_log;

      if(!check_not_nan(function, y, "Random variable", &cdf_log))
        return cdf_log;
      if(!check_finite(function, mu, "Location parameter", &cdf_log))
        return cdf_log;
      if(!check_finite(function, sigma, "Scale parameter", &cdf_log))
        return cdf_log;
      if(!check_positive(function, sigma, "Scale parameter", &cdf_log))
        return cdf_log;
      if (!(check_consistent_sizes(function, y, mu, sigma,
                                   "Random variable", "Location parameter", 
                                   "Scale Parameter", &cdf_log)))
        return cdf_log;
      
      using std::log;
      using std::exp;

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      const double log_half = std::log(0.5);
      size_t N = max_size(y, mu, sigma);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / sigma_dbl;
        const double inv_sigma = 1.0 / sigma_dbl;
        if(y_dbl < mu_dbl) {
          // log cdf
          cdf_log += log_half + scaled_diff;

          // gradients
          if (!is_constant_struct<T_y>::value)
            operands_and_partials.d_x1[n] += inv_sigma;
          if (!is_constant_struct<T_loc>::value)
            operands_and_partials.d_x2[n] -= inv_sigma;
          if (!is_constant_struct<T_scale>::value)
            operands_and_partials.d_x3[n] -= scaled_diff * inv_sigma;
        }
        else {
          //log cdf
          cdf_log += log(1.0 - 0.5 * exp(-scaled_diff));

          //gradients
          const double rep_deriv = 1.0 / (2.0 * exp(scaled_diff) - 1.0);
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv * inv_sigma;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv * inv_sigma;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff 
            * inv_sigma;
        }
      }
      return operands_and_partials.to_var(cdf_log);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    double_exponential_ccdf_log(const T_y& y, const T_loc& mu, 
                               const T_scale& sigma) {
      static const char* function
        = "stan::prob::double_exponential_ccdf_log(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_positive;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      double ccdf_log(0.0);

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return ccdf_log;

      if(!check_not_nan(function, y, "Random variable", &ccdf_log))
        return ccdf_log;
      if(!check_finite(function, mu, "Location parameter", &ccdf_log))
        return ccdf_log;
      if(!check_finite(function, sigma, "Scale parameter", &ccdf_log))
        return ccdf_log;
      if(!check_positive(function, sigma, "Scale parameter", &ccdf_log))
        return ccdf_log;
      if (!(check_consistent_sizes(function, y, mu, sigma,
                                   "Random variable", "Location parameter", 
                                   "Scale Parameter", &ccdf_log)))
        return ccdf_log;
      
      using std::log;
      using std::exp;

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      const double log_half = std::log(0.5);
      size_t N = max_size(y, mu, sigma);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / sigma_dbl;
        const double inv_sigma = 1.0 / sigma_dbl;
        if(y_dbl < mu_dbl) {
          //log ccdf
          ccdf_log += log(1.0 - 0.5 * exp(scaled_diff));

          //gradients
          const double rep_deriv = 1.0 / (2.0 * exp(-scaled_diff) - 1.0);
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv * inv_sigma;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += rep_deriv * inv_sigma;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += rep_deriv * scaled_diff 
            * inv_sigma;
        }
        else {
          // log ccdf
          ccdf_log += log_half - scaled_diff;

          // gradients
          if (!is_constant_struct<T_y>::value)
            operands_and_partials.d_x1[n] -= inv_sigma;
          if (!is_constant_struct<T_loc>::value)
            operands_and_partials.d_x2[n] += inv_sigma;
          if (!is_constant_struct<T_scale>::value)
            operands_and_partials.d_x3[n] += scaled_diff * inv_sigma;
        }
      }
      return operands_and_partials.to_var(ccdf_log);
    }
    
    template <class RNG>
    inline double
    double_exponential_rng(const double mu,
                           const double sigma,
                           RNG& rng) {
      using boost::variate_generator;
      using boost::random::uniform_01;
      using std::log;
      using std::abs;

      static const char* function
        = "stan::prob::double_exponential_rng(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_positive;

      if(!check_finite(function, mu, "Location parameter"))
        return 0;
      if(!check_finite(function, sigma, "Scale parameter"))
        return 0;
      if(!check_positive(function, sigma, "Scale parameter"))
        return 0;

      variate_generator<RNG&, uniform_01<> >
        rng_unit_01(rng, uniform_01<>());
      double a = 0;
      double laplaceRN = rng_unit_01();
      if(0.5 - laplaceRN > 0)
        a = 1.0;
      else if(0.5 - laplaceRN < 0)
        a = -1.0;
      return mu - sigma * a * log(1 - 2 * abs(0.5 - laplaceRN));
    }
  }
}
#endif
