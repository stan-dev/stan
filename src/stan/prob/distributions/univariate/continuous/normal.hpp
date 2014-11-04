#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__NORMAL_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__NORMAL_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/utility/enable_if.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the normal density for the specified scalar(s) given
     * the specified mean(s) and deviation(s). y, mu, or sigma can
     * each be either a scalar or a std::vector. Any vector inputs
     * must be the same length.
     *
     * <p>The result log probability is defined to be the sum of the
     * log probabilities for each observation/mean/deviation triple.
     * @param y (Sequence of) scalar(s).
     * @param mu (Sequence of) location parameter(s)
     * for the normal distribution.
     * @param sigma (Sequence of) scale parameters for the normal
     * distribution.
     * @return The log of the product of the densities.
     * @throw std::domain_error if the scale is not positive.
     * @tparam T_y Underlying type of scalar in sequence.
     * @tparam T_loc Type of location parameter.
     */
    template <bool propto, 
              typename T_y, typename T_loc, typename T_scale>
    typename boost::enable_if_c<is_var_or_arithmetic<T_y,T_loc,T_scale>::value,
                                typename return_type<T_y,T_loc,T_scale>::type>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::normal_log");

      using std::log;
      using stan::is_constant_struct;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;

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
      check_positive(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      DoubleVectorView<true,is_vector<T_scale>::value> inv_sigma(length(sigma));
      DoubleVectorView<include_summand<propto,T_scale>::value,is_vector<T_scale>::value> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
        if (include_summand<propto,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
      
        // reusable subexpression values
        const double y_minus_mu_over_sigma 
          = (y_dbl - mu_dbl) * inv_sigma[n];
        const double y_minus_mu_over_sigma_squared 
          = y_minus_mu_over_sigma * y_minus_mu_over_sigma;

        static double NEGATIVE_HALF = - 0.5;

        // log probability
        if (include_summand<propto>::value)
          logp += NEG_LOG_SQRT_TWO_PI;
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp += NEGATIVE_HALF * y_minus_mu_over_sigma_squared;

        // gradients
        double scaled_diff = inv_sigma[n] * y_minus_mu_over_sigma;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= scaled_diff;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += scaled_diff;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += -inv_sigma[n] + inv_sigma[n] * y_minus_mu_over_sigma_squared;
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return normal_log<false>(y,mu,sigma);
    }

       /**
     * Calculates the normal cumulative distribution function for the given
     * variate, location, and scale.
     * 
     * \f$\Phi(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\inf}^x e^{-t^2/2} dt\f$.
     * 
     * Errors are configured by policy.  All variables must be finite
     * and the scale must be strictly greater than zero.
     * 
     * @param y A scalar variate.
     * @param mu The location of the normal distribution.
     * @param sigma The scale of the normal distriubtion
     * @return The unit normal cdf evaluated at the specified arguments.
     * @tparam T_y Type of y.
     * @tparam T_loc Type of mean parameter.
     * @tparam T_scale Type of standard deviation paramater.
     * @tparam Policy Error-handling policy.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::normal_cdf");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::INV_SQRT_2;

      double cdf(1.0);

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);
                             

     agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
       operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / (sigma_dbl * SQRT_2);
        double cdf_;
        if (scaled_diff < -37.5 * INV_SQRT_2)
          cdf_ = 0.0;
        else if (scaled_diff < -5.0 * INV_SQRT_2)
          cdf_ = 0.5 * erfc(-scaled_diff);
        else if (scaled_diff > 8.25 * INV_SQRT_2)
          cdf_ = 1;
        else
          cdf_ = 0.5 * (1.0 + erf(scaled_diff));

        // cdf
        cdf *= cdf_;

        // gradients
        const double rep_deriv = SQRT_TWO_OVER_PI * 0.5 
          * exp(-scaled_diff * scaled_diff) / cdf_ / sigma_dbl;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff * SQRT_2;
      }

      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= cdf;
      if (!is_constant_struct<T_loc>::value)
        for (size_t n = 0; n < stan::length(mu); ++n) 
          operands_and_partials.d_x2[n] *= cdf;
      if (!is_constant_struct<T_scale>::value)
        for (size_t n = 0; n < stan::length(sigma); ++n) 
          operands_and_partials.d_x3[n] *= cdf;

      return operands_and_partials.to_var(cdf);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::normal_cdf_log");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::INV_SQRT_2;

      double cdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf_log;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      double log_half = std::log(0.5);  
    
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);

        const double scaled_diff = (y_dbl - mu_dbl) / (sigma_dbl * SQRT_2);
        
        double one_p_erf;
        if (scaled_diff < -37.5 * INV_SQRT_2)
          one_p_erf = 0.0;
        else if (scaled_diff < -5.0 * INV_SQRT_2)
          one_p_erf =  erfc(-scaled_diff);
        else if (scaled_diff > 8.25 * INV_SQRT_2)
          one_p_erf = 2.0;
        else
          one_p_erf = 1.0 + erf(scaled_diff);

        // log cdf
        cdf_log += log_half + log(one_p_erf);

        // gradients
        const double rep_deriv = SQRT_TWO_OVER_PI 
          * exp(-scaled_diff * scaled_diff) / one_p_erf;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv / sigma_dbl;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv / sigma_dbl;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff 
            * stan::math::SQRT_2 / sigma_dbl;
      }
      return operands_and_partials.to_var(cdf_log);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    normal_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::normal_ccdf_log");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::INV_SQRT_2;

      double ccdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return ccdf_log;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      double log_half = std::log(0.5);  
    
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);

        const double scaled_diff = (y_dbl - mu_dbl) / (sigma_dbl * SQRT_2);
        
        double one_m_erf;
        if (scaled_diff < -37.5 * INV_SQRT_2)
          one_m_erf = 2.0;
        else if (scaled_diff < -5.0 * INV_SQRT_2)
          one_m_erf =  2.0 - erfc(-scaled_diff);
        else if (scaled_diff > 8.25 * INV_SQRT_2)
          one_m_erf = 0.0;
        else
          one_m_erf = 1.0 - erf(scaled_diff);

        // log ccdf
        ccdf_log += log_half + log(one_m_erf);

        // gradients
        const double rep_deriv = SQRT_TWO_OVER_PI 
          * exp(-scaled_diff * scaled_diff) / one_m_erf;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv / sigma_dbl;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += rep_deriv / sigma_dbl;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += rep_deriv * scaled_diff 
            * stan::math::SQRT_2 / sigma_dbl;
      }
      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    normal_rng(const double mu,
               const double sigma,
               RNG& rng) {
      using boost::variate_generator;
      using boost::normal_distribution;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;

      static const std::string function("stan::prob::normal_rng");

      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Location parameter", mu);
      check_positive(function, "Scale parameter", sigma);
      check_not_nan(function, "Scale parameter", sigma);

      variate_generator<RNG&, normal_distribution<> >
        norm_rng(rng, normal_distribution<>(mu, sigma));
      return norm_rng();
    }
  }
}
#endif
