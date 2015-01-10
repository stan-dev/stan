#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__LOGNORMAL_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__LOGNORMAL_HPP

#include <boost/random/lognormal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/functions/constants.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/functions/square.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    // LogNormal(y|mu,sigma)  [y >= 0;  sigma > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::lognormal_log");
      typedef typename stan::partials_return_type<T_y,T_loc,T_scale>::type
        T_partials_return;

      using stan::is_constant_struct;
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;      
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
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);
      
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      
      for (size_t n = 0; n < length(y); n++)
        if (value_of(y_vec[n]) <= 0)
          return LOG_ZERO;
      
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);
 
      using stan::math::square;
      using std::log;
      using stan::prob::NEG_LOG_SQRT_TWO_PI;
      

      VectorBuilder<include_summand<propto,T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      if (include_summand<propto, T_scale>::value)
        for (size_t n = 0; n < length(sigma); n++)
          log_sigma[n] = log(value_of(sigma_vec[n]));

      VectorBuilder<include_summand<propto,T_y,T_loc,T_scale>::value,
                    T_partials_return, T_scale> inv_sigma(length(sigma));
      VectorBuilder<include_summand<propto,T_y,T_loc,T_scale>::value,
                    T_partials_return, T_scale> inv_sigma_sq(length(sigma));
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        for (size_t n = 0; n < length(sigma); n++)
          inv_sigma[n] = 1 / value_of(sigma_vec[n]);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        for (size_t n = 0; n < length(sigma); n++)
          inv_sigma_sq[n] = inv_sigma[n] * inv_sigma[n];
      
      VectorBuilder<include_summand<propto,T_y,T_loc,T_scale>::value,
                    T_partials_return, T_y> log_y(length(y));
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        for (size_t n = 0; n < length(y); n++)
          log_y[n] = log(value_of(y_vec[n]));

      VectorBuilder<!is_constant_struct<T_y>::value,
                    T_partials_return, T_y> inv_y(length(y));
      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < length(y); n++)
          inv_y[n] = 1 / value_of(y_vec[n]);

      if (include_summand<propto>::value)
        logp += N * NEG_LOG_SQRT_TWO_PI;

      for (size_t n = 0; n < N; n++) {
        const T_partials_return mu_dbl = value_of(mu_vec[n]);

        T_partials_return logy_m_mu(0);
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logy_m_mu = log_y[n] - mu_dbl;

        T_partials_return logy_m_mu_sq = logy_m_mu * logy_m_mu;
        T_partials_return logy_m_mu_div_sigma(0);
        if (contains_nonconstant_struct<T_y,T_loc,T_scale>::value)
          logy_m_mu_div_sigma = logy_m_mu * inv_sigma_sq[n];
  

        // log probability
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y>::value)
          logp -= log_y[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp -= 0.5 * logy_m_mu_sq * inv_sigma_sq[n];

        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= (1 + logy_m_mu_div_sigma) * inv_y[n];
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += logy_m_mu_div_sigma;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += (logy_m_mu_div_sigma * logy_m_mu - 1) * inv_sigma[n];
      }
      return operands_and_partials.to_var(logp,y,mu,sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    lognormal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return lognormal_log<false>(y,mu,sigma);
    }


    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    lognormal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::lognormal_cdf");

      typedef typename stan::partials_return_type<T_y,T_loc,T_scale>::type 
        T_partials_return;

      T_partials_return cdf = 1.0;
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive_finite;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      const double sqrt_pi = std::sqrt(stan::math::pi());

      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0.0) 
          return operands_and_partials.to_var(0.0,y,mu,sigma);
      }

      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return scaled_diff = (log(y_dbl) - mu_dbl) 
          / (sigma_dbl * SQRT_2);
        const T_partials_return rep_deriv = SQRT_2 * 0.5 / sqrt_pi 
          * exp(-scaled_diff * scaled_diff) / sigma_dbl;

        //cdf
        const T_partials_return cdf_ = 0.5 * erfc(-scaled_diff);
        cdf *= cdf_;

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv / cdf_ / y_dbl ;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv / cdf_ ;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff * SQRT_2 
            / cdf_;
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

      return operands_and_partials.to_var(cdf,y,mu,sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    lognormal_cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::lognormal_cdf_log");
      typedef typename stan::partials_return_type<T_y,T_loc,T_scale>::type 
        T_partials_return;

      T_partials_return cdf_log = 0.0;
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive_finite;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return cdf_log;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      const double sqrt_pi = std::sqrt(stan::math::pi());

      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0.0) 
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              y,mu,sigma);
      }

      const double log_half = std::log(0.5);

      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return scaled_diff = (log(y_dbl) - mu_dbl)
          / (sigma_dbl * SQRT_2);
        const T_partials_return rep_deriv = SQRT_2 / sqrt_pi 
          * exp(-scaled_diff * scaled_diff) / sigma_dbl;

        //cdf_log
        const T_partials_return erfc_calc = erfc(-scaled_diff);
        cdf_log += log_half + log(erfc_calc);

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv / erfc_calc / y_dbl ;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv / erfc_calc;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff * SQRT_2 
            / erfc_calc;
      }

      return operands_and_partials.to_var(cdf_log,y,mu,sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    lognormal_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      static const std::string function("stan::prob::lognormal_ccdf_log");
      typedef typename stan::partials_return_type<T_y,T_loc,T_scale>::type 
        T_partials_return;

      T_partials_return ccdf_log = 0.0;
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive_finite;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return ccdf_log;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      const double sqrt_pi = std::sqrt(stan::math::pi());

      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0.0) 
          return operands_and_partials.to_var(0.0,y,mu,sigma);
      }

      const double log_half = std::log(0.5);

      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return scaled_diff = (log(y_dbl) - mu_dbl) 
          / (sigma_dbl * SQRT_2);
        const T_partials_return rep_deriv = SQRT_2 / sqrt_pi 
          * exp(-scaled_diff * scaled_diff) / sigma_dbl;

        //ccdf_log
        const T_partials_return erfc_calc = erfc(scaled_diff);
        ccdf_log += log_half + log(erfc_calc);

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv / erfc_calc / y_dbl ;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += rep_deriv / erfc_calc;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += rep_deriv * scaled_diff * SQRT_2 
            / erfc_calc;
      }

      return operands_and_partials.to_var(ccdf_log,y,mu,sigma);
    }


    template <class RNG>
    inline double
    lognormal_rng(const double mu,
                  const double sigma,
                  RNG& rng) {
      using boost::variate_generator;
      using boost::random::lognormal_distribution;

      static const std::string function("stan::prob::lognormal_rng");

      using stan::math::check_finite;
      using stan::math::check_positive_finite;

      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      variate_generator<RNG&, lognormal_distribution<> >
        lognorm_rng(rng, lognormal_distribution<>(mu, sigma));
      return lognorm_rng();
    }
  }
}
#endif
