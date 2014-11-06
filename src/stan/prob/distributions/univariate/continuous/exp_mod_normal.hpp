#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__EXP__MOD__NORMAL__HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__EXP__MOD__NORMAL__HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>
#include <stan/prob/traits.hpp>
#include <stan/math/functions/value_of.hpp>

namespace stan {

  namespace prob {

    template <bool propto, 
              typename T_y, typename T_loc, typename T_scale,
              typename T_inv_scale>
    typename return_type<T_y,T_loc,T_scale, T_inv_scale>::type
    exp_mod_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                       const T_inv_scale& lambda) {
      static const std::string function("stan::prob::exp_mod_normal_log");

      using stan::is_constant_struct;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(lambda)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Inv_scale paramter", lambda);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale,T_inv_scale>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale> 
        operands_and_partials(y, mu, sigma,lambda);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);

      for (size_t n = 0; n < N; n++) {
        //pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);

        const double pi_dbl = boost::math::constants::pi<double>();

        // log probability
        if (include_summand<propto>::value)
          logp -= log(2.0);
        if (include_summand<propto, T_inv_scale>::value)
          logp += log(lambda_dbl);
        if (include_summand<propto,T_y,T_loc,T_scale,T_inv_scale>::value)
          logp += lambda_dbl
            * (mu_dbl + 0.5 * lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) 
            + log(boost::math::erfc((mu_dbl + lambda_dbl * sigma_dbl 
                                     * sigma_dbl - y_dbl) 
                                    / (std::sqrt(2.0) * sigma_dbl)));

        // gradients
        const double deriv_logerfc 
          = -2.0 / std::sqrt(pi_dbl) 
          * exp(-(mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) 
                / (std::sqrt(2.0) * sigma_dbl) 
                * (mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) 
                / (sigma_dbl * std::sqrt(2.0))) 
          / boost::math::erfc((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl 
                               - y_dbl)
                              / (sigma_dbl * std::sqrt(2.0)));

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += -lambda_dbl 
            + deriv_logerfc * -1.0 / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] 
            += lambda_dbl 
            + deriv_logerfc / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += sigma_dbl * lambda_dbl * lambda_dbl 
            + deriv_logerfc 
            * (-mu_dbl / (sigma_dbl * sigma_dbl * std::sqrt(2.0)) 
               + lambda_dbl / std::sqrt(2.0) 
               + y_dbl / (sigma_dbl * sigma_dbl * std::sqrt(2.0)));
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x4[n]
            += 1 / lambda_dbl + lambda_dbl * sigma_dbl * sigma_dbl 
            + mu_dbl - y_dbl + deriv_logerfc * sigma_dbl / std::sqrt(2.0);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale, 
              typename T_inv_scale>
    inline
    typename return_type<T_y,T_loc,T_scale, T_inv_scale>::type
    exp_mod_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                       const T_inv_scale& lambda) {
      return exp_mod_normal_log<false>(y,mu,sigma,lambda);
    }

    template <typename T_y, typename T_loc, typename T_scale, 
              typename T_inv_scale>
    typename return_type<T_y,T_loc,T_scale,T_inv_scale>::type
    exp_mod_normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                       const T_inv_scale& lambda) {
      static const std::string function("stan::prob::exp_mod_normal_cdf");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double cdf(1.0);
      //check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(lambda)))
        return cdf;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive_finite(function, "Scale parameter", sigma);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_not_nan(function, "Inv_scale parameter", lambda);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Inv_scale paramter", lambda);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale> 
        operands_and_partials(y, mu, sigma,lambda);

      using stan::math::SQRT_2;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);
      const double sqrt_pi = std::sqrt(stan::math::pi());
      for (size_t n = 0; n < N; n++) {

        if(boost::math::isinf(y_vec[n])) {
          if (y_vec[n] < 0.0)
            return operands_and_partials.to_var(0.0);
        }

        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);
        const double u = lambda_dbl * (y_dbl - mu_dbl);
        const double v = lambda_dbl * sigma_dbl ;
        const double v_sq = v * v;
        const double scaled_diff = (y_dbl - mu_dbl) / (SQRT_2 * sigma_dbl);
        const double scaled_diff_sq = scaled_diff * scaled_diff;
        const double erf_calc = 0.5 * (1 + erf(-v / SQRT_2 + scaled_diff));
        const double deriv_1 = lambda_dbl * exp(0.5 * v_sq - u) * erf_calc;
        const double deriv_2 = SQRT_2 / sqrt_pi * 0.5 * exp(0.5 * v_sq 
                                                            - (scaled_diff 
                                - (v / SQRT_2)) * (scaled_diff 
                                - (v / SQRT_2)) - u) / sigma_dbl;
        const double deriv_3 = SQRT_2 / sqrt_pi * 0.5 * exp(-scaled_diff_sq) 
          / sigma_dbl;

        const double cdf_ = 0.5 * (1 + erf(u / (v * SQRT_2))) 
          - exp(-u + v_sq * 0.5) * (erf_calc);

          cdf *= cdf_;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += (deriv_1 - deriv_2 + deriv_3) 
            / cdf_;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += (-deriv_1 + deriv_2 - deriv_3) 
            / cdf_;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += (-deriv_1 * v - deriv_3 
            * scaled_diff * SQRT_2 - deriv_2 * sigma_dbl * SQRT_2 * (-SQRT_2 
            * 0.5 * (-lambda_dbl + scaled_diff * SQRT_2 / sigma_dbl) - SQRT_2 
            * lambda_dbl)) / cdf_;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x4[n] += exp(0.5 * v_sq - u) * (SQRT_2 
            / sqrt_pi * 0.5 * sigma_dbl * exp(-(v / SQRT_2 - scaled_diff) * (v
            / SQRT_2 - scaled_diff)) - (v * sigma_dbl + mu_dbl - y_dbl) 
            * erf_calc) / cdf_;
      }

      if (!is_constant_struct<T_y>::value) {
        for(size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= cdf;
      }
      if (!is_constant_struct<T_loc>::value) {
        for(size_t n = 0; n < stan::length(mu); ++n) 
          operands_and_partials.d_x2[n] *= cdf;
      }
      if (!is_constant_struct<T_scale>::value) {
        for(size_t n = 0; n < stan::length(sigma); ++n) 
          operands_and_partials.d_x3[n] *= cdf;
      }
      if (!is_constant_struct<T_inv_scale>::value) {
        for(size_t n = 0; n < stan::length(lambda); ++n) 
          operands_and_partials.d_x4[n] *= cdf;
      }

      return operands_and_partials.to_var(cdf);
    }

    template <typename T_y, typename T_loc, typename T_scale, 
              typename T_inv_scale>
    typename return_type<T_y,T_loc,T_scale,T_inv_scale>::type
    exp_mod_normal_cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                       const T_inv_scale& lambda) {
      static const std::string function("stan::prob::exp_mod_normal_cdf_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double cdf_log(0.0);
      //check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(lambda)))
        return cdf_log;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive_finite(function, "Scale parameter", sigma);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_not_nan(function, "Inv_scale parameter", lambda);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Inv_scale paramter", lambda);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale> 
        operands_and_partials(y, mu, sigma,lambda);

      using stan::math::SQRT_2;
      using std::log;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);
      const double sqrt_pi = std::sqrt(stan::math::pi());
      for (size_t n = 0; n < N; n++) {

        if(boost::math::isinf(y_vec[n])) {
          if (y_vec[n] < 0.0)
            return operands_and_partials.to_var(stan::math::negative_infinity());
          else
            return operands_and_partials.to_var(0.0);
        }

        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);
        const double u = lambda_dbl * (y_dbl - mu_dbl);
        const double v = lambda_dbl * sigma_dbl ;
        const double v_sq = v * v;
        const double scaled_diff = (y_dbl - mu_dbl) / (SQRT_2 * sigma_dbl);
        const double scaled_diff_sq = scaled_diff * scaled_diff;
        const double erf_calc1 = 0.5 * (1 + erf(u / (v * SQRT_2)));
        const double erf_calc2 = 0.5 * (1 + erf(u / (v * SQRT_2) - v / SQRT_2));
        const double deriv_1 = lambda_dbl * exp(0.5 * v_sq - u) * erf_calc2;
        const double deriv_2 = SQRT_2 / sqrt_pi * 0.5 
          * exp(0.5 * v_sq - (-scaled_diff + (v / SQRT_2)) 
                * (-scaled_diff + (v / SQRT_2)) - u) / sigma_dbl;
        const double deriv_3 = SQRT_2 / sqrt_pi * 0.5 * exp(-scaled_diff_sq) 
          / sigma_dbl;

        const double denom = erf_calc1 - erf_calc2 * exp(0.5 * v_sq - u);
        const double cdf_ = erf_calc1 - exp(-u + v_sq * 0.5) * (erf_calc2);

        cdf_log += log(cdf_);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += (deriv_1 - deriv_2 + deriv_3) 
                                             / denom;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += (-deriv_1 + deriv_2 - deriv_3) 
                                             / denom;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += (-deriv_1 * v - deriv_3 * scaled_diff 
                * SQRT_2 - deriv_2 * sigma_dbl * SQRT_2 
                * (-SQRT_2 * 0.5 * (-lambda_dbl + scaled_diff * SQRT_2
                                    / sigma_dbl) 
                   - SQRT_2 * lambda_dbl))
            / denom;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x4[n] 
            += exp(0.5 * v_sq - u) 
            * (SQRT_2 / sqrt_pi * 0.5 * sigma_dbl 
               * exp(-(v / SQRT_2 - scaled_diff) 
                     * (v / SQRT_2 - scaled_diff)) 
               - (v * sigma_dbl + mu_dbl - y_dbl) * erf_calc2) 
            / denom;
      }

      return operands_and_partials.to_var(cdf_log);
    }

    template <typename T_y, typename T_loc, typename T_scale, 
              typename T_inv_scale>
    typename return_type<T_y,T_loc,T_scale,T_inv_scale>::type
    exp_mod_normal_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                       const T_inv_scale& lambda) {
      static const std::string function("stan::prob::exp_mod_normal_ccdf_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double ccdf_log(0.0);
      //check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(lambda)))
        return ccdf_log;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive_finite(function, "Scale parameter", sigma);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_not_nan(function, "Inv_scale parameter", lambda);
      check_consistent_sizes(function,
                             "Random variable",y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Inv_scale paramter", lambda);
      

      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale> 
        operands_and_partials(y, mu, sigma,lambda);

      using stan::math::SQRT_2;
      using std::log;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);
      const double sqrt_pi = std::sqrt(stan::math::pi());
      for (size_t n = 0; n < N; n++) {

        if(boost::math::isinf(y_vec[n])) {
          if (y_vec[n] > 0.0)
            return operands_and_partials.to_var(stan::math::negative_infinity());
          else
            return operands_and_partials.to_var(0.0);
        }

        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);
        const double u = lambda_dbl * (y_dbl - mu_dbl);
        const double v = lambda_dbl * sigma_dbl ;
        const double v_sq = v * v;
        const double scaled_diff = (y_dbl - mu_dbl) / (SQRT_2 * sigma_dbl);
        const double scaled_diff_sq = scaled_diff * scaled_diff;
        const double erf_calc1 = 0.5 * (1 + erf(u / (v * SQRT_2)));
        const double erf_calc2 = 0.5 * (1 + erf(u / (v * SQRT_2) - v / SQRT_2));

        const double deriv_1 = lambda_dbl * exp(0.5 * v_sq - u) * erf_calc2;
        const double deriv_2 = SQRT_2 / sqrt_pi * 0.5 
          * exp(0.5 * v_sq 
                - (-scaled_diff + (v / SQRT_2)) * (-scaled_diff  
                                                   + (v / SQRT_2)) - u) 
          / sigma_dbl;
        const double deriv_3 = SQRT_2 / sqrt_pi * 0.5 * exp(-scaled_diff_sq) / sigma_dbl;

        const double ccdf_ = 1.0 - erf_calc1 + exp(-u + v_sq * 0.5) * (erf_calc2);

        ccdf_log += log(ccdf_);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n]
            -= (deriv_1 - deriv_2 + deriv_3) / ccdf_;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] 
            -= (-deriv_1 + deriv_2 - deriv_3) / ccdf_;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            -= (-deriv_1 * v - deriv_3 * scaled_diff * SQRT_2 - deriv_2 
                * sigma_dbl * SQRT_2 
                * (-SQRT_2 * 0.5 * (-lambda_dbl + scaled_diff * SQRT_2 
                                    / sigma_dbl)
                   - SQRT_2 * lambda_dbl)) 
            / ccdf_;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x4[n] -= exp(0.5 * v_sq - u) 
            * (SQRT_2 / sqrt_pi * 0.5 * sigma_dbl 
               * exp(-(v / SQRT_2 - scaled_diff) * (v / SQRT_2 - scaled_diff)) 
               - (v * sigma_dbl + mu_dbl - y_dbl) * erf_calc2) 
            / ccdf_;
      }

      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    exp_mod_normal_rng(const double mu,
                       const double sigma,
                       const double lambda,
                       RNG& rng) {

      static const std::string function("stan::prob::exp_mod_normal_rng");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;

      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_positive_finite(function, "Scale parameter", sigma);

      return stan::prob::normal_rng(mu, sigma,rng) + stan::prob::exponential_rng(lambda, rng);
    }
  }
}
#endif



