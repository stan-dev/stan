#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__NORMAL_HPP__

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/utility/enable_if.hpp>

#include <stan/agrad.hpp>
#include <stan/math.hpp>
#include <stan/math/error_handling.hpp>
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
      static const char* function = "stan::prob::normal_log(%1%)";

      using std::log;
      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
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
      if (!check_not_nan(function, y, "Random variable", &logp))
        return logp;
      if (!check_finite(function, mu, "Location parameter", 
                        &logp))
        return logp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,
                                   "Random variable","Location parameter","Scale parameter",
                                   &logp)))
        return logp;

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
      static const char* function = "stan::prob::normal_cdf(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      double cdf(1.0);

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf;

      if (!check_not_nan(function, y, "Random variable", &cdf))
        return cdf;
      if (!check_finite(function, mu, "Location parameter", &cdf))
        return cdf;
      if (!check_not_nan(function, sigma, "Scale parameter", 
                         &cdf))
        return cdf;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &cdf))
        return cdf;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,
                                   "Random variable","Location parameter",
                                   "Scale parameter",
                                   &cdf)))
        return cdf;

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
        const double cdf_ = 0.5 * (1.0 + erf(scaled_diff));

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
      static const char* function = "stan::prob::normal_cdf_log(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      double cdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf_log;

      if (!check_not_nan(function, y, "Random variable", &cdf_log))
        return cdf_log;
      if (!check_finite(function, mu, "Location parameter", &cdf_log))
        return cdf_log;
      if (!check_not_nan(function, sigma, "Scale parameter", 
                         &cdf_log))
        return cdf_log;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &cdf_log))
        return cdf_log;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,
                                   "Random variable","Location parameter",
                                   "Scale parameter", &cdf_log)))
        return cdf_log;

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
        
        const double one_p_erf = 1.0 + erf(scaled_diff);
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
      static const char* function = "stan::prob::normal_ccdf_log(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      double ccdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return ccdf_log;

      if (!check_not_nan(function, y, "Random variable", &ccdf_log))
        return ccdf_log;
      if (!check_finite(function, mu, "Location parameter", &ccdf_log))
        return ccdf_log;
      if (!check_not_nan(function, sigma, "Scale parameter", 
                         &ccdf_log))
        return ccdf_log;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &ccdf_log))
        return ccdf_log;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,
                                   "Random variable","Location parameter",
                                   "Scale parameter", &ccdf_log)))
        return ccdf_log;

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
        
        const double one_m_erf = 1.0 - erf(scaled_diff);
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
      variate_generator<RNG&, normal_distribution<> >
        norm_rng(rng, normal_distribution<>(mu, sigma));
      return norm_rng();
    }
  }
}
#endif
