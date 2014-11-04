#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__SKEW__NORMAL__HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__SKEW__NORMAL__HPP

#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/math/functions/owens_t.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    template <bool propto, 
              typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y,T_loc,T_scale,T_shape>::type
    skew_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                    const T_shape& alpha) {
      static const std::string function("stan::prob::skew_normal_log");

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
            && stan::length(sigma)
            && stan::length(alpha)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_finite(function, "Shape parameter", alpha);
      check_positive(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Shape paramter", alpha);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale,T_shape>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_shape> 
        operands_and_partials(y, mu, sigma, alpha);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, sigma, alpha);

      DoubleVectorView<true,is_vector<T_scale>::value> inv_sigma(length(sigma));
      DoubleVectorView<include_summand<propto,T_scale>::value,
        is_vector<T_scale>::value> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
        if (include_summand<propto,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);

        // reusable subexpression values
        const double y_minus_mu_over_sigma 
          = (y_dbl - mu_dbl) * inv_sigma[n];
        const double pi_dbl = boost::math::constants::pi<double>();

        // log probability
        if (include_summand<propto>::value)
          logp -=  0.5 * log(2.0 * pi_dbl);
        if (include_summand<propto, T_scale>::value)
          logp -= log(sigma_dbl);
        if (include_summand<propto,T_y, T_loc, T_scale>::value)
          logp -= y_minus_mu_over_sigma * y_minus_mu_over_sigma / 2.0;
        if (include_summand<propto,T_y,T_loc,T_scale,T_shape>::value)
          logp += log(boost::math::erfc(-alpha_dbl * y_minus_mu_over_sigma 
                                        / std::sqrt(2.0)));

        // gradients
        double deriv_logerf 
          = 2.0 / std::sqrt(pi_dbl) 
          * exp(-alpha_dbl * y_minus_mu_over_sigma / std::sqrt(2.0) 
                * alpha_dbl * y_minus_mu_over_sigma / std::sqrt(2.0))
          / (1 + boost::math::erf(alpha_dbl * y_minus_mu_over_sigma 
                                  / std::sqrt(2.0)));
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += -y_minus_mu_over_sigma / sigma_dbl 
            + deriv_logerf * alpha_dbl / (sigma_dbl * std::sqrt(2.0)) ;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] 
            += y_minus_mu_over_sigma / sigma_dbl 
            + deriv_logerf * -alpha_dbl / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += -1.0 / sigma_dbl 
            + y_minus_mu_over_sigma * y_minus_mu_over_sigma / sigma_dbl 
            - deriv_logerf * y_minus_mu_over_sigma * alpha_dbl 
              / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] 
            += deriv_logerf * y_minus_mu_over_sigma / std::sqrt(2.0);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline
    typename return_type<T_y,T_loc,T_scale, T_shape>::type
    skew_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
                    const T_shape& alpha) {
      return skew_normal_log<false>(y,mu,sigma,alpha);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y,T_loc,T_scale,T_shape>::type
    skew_normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                    const T_shape& alpha) {
      static const std::string function("stan::prob::skew_normal_cdf");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::owens_t;
      using stan::math::value_of;

      double cdf(1.0);
      
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(alpha)))
        return cdf;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive(function, "Scale parameter", sigma);
      check_finite(function, "Shape parameter", alpha);
      check_not_nan(function, "Shape parameter", alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Shape paramter", alpha);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_shape> 
        operands_and_partials(y, mu, sigma, alpha);

      using stan::math::SQRT_2;
      using stan::math::pi;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, sigma, alpha);
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double alpha_dbl_sq = alpha_dbl * alpha_dbl;
        const double diff = (y_dbl - mu_dbl) / sigma_dbl;
        const double diff_sq = diff * diff;
        const double scaled_diff =  diff / SQRT_2;
        const double scaled_diff_sq =  diff_sq * 0.5;
        const double cdf_ = 0.5 * erfc(-scaled_diff) - 2 * owens_t(diff, 
                                                                 alpha_dbl);
        //cdf
        cdf *= cdf_;

        //gradients
        const double deriv_erfc = SQRT_TWO_OVER_PI * 0.5 * exp(-scaled_diff_sq)
          / sigma_dbl;
        const double deriv_owens = erf(alpha_dbl * scaled_diff) 
          * exp(-scaled_diff_sq) / SQRT_TWO_OVER_PI / (-2.0 * pi()) / sigma_dbl;
        const double rep_deriv = (-2.0 * deriv_owens + deriv_erfc) / cdf_;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * diff;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] += -2.0 * exp(-0.5 * diff_sq * (1.0 
                   + alpha_dbl_sq)) / ((1 + alpha_dbl_sq) * 2.0 * pi()) / cdf_;
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
      if (!is_constant_struct<T_shape>::value)
        for (size_t n = 0; n < stan::length(alpha); ++n) 
          operands_and_partials.d_x4[n] *= cdf;

      return operands_and_partials.to_var(cdf);    
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y,T_loc,T_scale,T_shape>::type
    skew_normal_cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                    const T_shape& alpha) {
      static const std::string function("stan::prob::skew_normal_cdf_log");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::owens_t;

      double cdf_log(0.0);
      
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(alpha)))
        return cdf_log;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive(function, "Scale parameter", sigma);
      check_finite(function, "Shape parameter", alpha);
      check_not_nan(function, "Shape parameter", alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Shape paramter", alpha);


      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_shape> 
        operands_and_partials(y, mu, sigma, alpha);

      using stan::math::SQRT_2;
      using stan::math::pi;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, sigma, alpha);
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double alpha_dbl_sq = alpha_dbl * alpha_dbl;
        const double diff = (y_dbl - mu_dbl) / sigma_dbl;
        const double diff_sq = diff * diff;
        const double scaled_diff =  diff / SQRT_2;
        const double scaled_diff_sq =  diff_sq * 0.5;
        const double cdf_log_ = 0.5 * erfc(-scaled_diff) - 2 * owens_t(diff, 
                                                                 alpha_dbl);
        //cdf_log
        cdf_log += log(cdf_log_);

        //gradients
        const double deriv_erfc = SQRT_TWO_OVER_PI * 0.5 * exp(-scaled_diff_sq)
          / sigma_dbl;
        const double deriv_owens = erf(alpha_dbl * scaled_diff) 
          * exp(-scaled_diff_sq) / SQRT_TWO_OVER_PI / (-2.0 * pi()) / sigma_dbl;
        const double rep_deriv = (-2.0 * deriv_owens + deriv_erfc) / cdf_log_;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * diff;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] += -2.0 * exp(-0.5 * diff_sq * (1.0
                + alpha_dbl_sq)) / ((1 + alpha_dbl_sq) * 2.0 * pi()) / cdf_log_;
      }

      return operands_and_partials.to_var(cdf_log);    
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y,T_loc,T_scale,T_shape>::type
    skew_normal_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                         const T_shape& alpha) {
      static const std::string function("stan::prob::skew_normal_ccdf_log");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::owens_t;
      using stan::math::value_of;

      double ccdf_log(0.0);
      
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(alpha)))
        return ccdf_log;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive(function, "Scale parameter", sigma);
      check_finite(function, "Shape parameter", alpha);
      check_not_nan(function, "Shape parameter", alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Shape paramter", alpha);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_shape> 
        operands_and_partials(y, mu, sigma, alpha);

      using stan::math::SQRT_2;
      using stan::math::pi;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, sigma, alpha);
      const double SQRT_TWO_OVER_PI = std::sqrt(2.0 / stan::math::pi());

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double alpha_dbl_sq = alpha_dbl * alpha_dbl;
        const double diff = (y_dbl - mu_dbl) / sigma_dbl;
        const double diff_sq = diff * diff;
        const double scaled_diff =  diff / SQRT_2;
        const double scaled_diff_sq =  diff_sq * 0.5;
        const double ccdf_log_ = 1.0 - 0.5 * erfc(-scaled_diff) + 2 * owens_t(diff, 
                                                                 alpha_dbl);
        //ccdf_log
        ccdf_log += log(ccdf_log_);

        //gradients
        const double deriv_erfc = SQRT_TWO_OVER_PI * 0.5 * exp(-scaled_diff_sq)
          / sigma_dbl;
        const double deriv_owens = erf(alpha_dbl * scaled_diff) 
          * exp(-scaled_diff_sq) / SQRT_TWO_OVER_PI / (-2.0 * pi()) / sigma_dbl;
        const double rep_deriv = (-2.0 * deriv_owens + deriv_erfc) / ccdf_log_;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += rep_deriv * diff;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n] -= -2.0 * exp(-0.5 * diff_sq * (1.0 
                   + alpha_dbl_sq)) / ((1 + alpha_dbl_sq) * 2.0 * pi()) / ccdf_log_;
      }

      return operands_and_partials.to_var(ccdf_log);    
    }

    template <class RNG>
    inline double
    skew_normal_rng(const double mu,
                    const double sigma,
                    const double alpha,
                    RNG& rng) {
      boost::math::skew_normal_distribution<>dist (mu, sigma, alpha);

      static const std::string function("stan::prob::skew_normal_rng");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;

      check_finite(function, "Location parameter", mu);
      check_finite(function, "Shape parameter", alpha);
      check_positive(function, "Scale parameter", sigma);

      return quantile(dist, stan::prob::uniform_rng(0.0,1.0,rng));
    }
  }
}
#endif

