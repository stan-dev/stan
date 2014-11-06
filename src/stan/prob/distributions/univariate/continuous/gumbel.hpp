#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__GUMBEL_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__GUMBEL_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/internal_math.hpp>
#include <stan/math/functions/value_of.hpp>

namespace stan {

  namespace prob {

    template <bool propto, typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    gumbel_log(const T_y& y, const T_loc& mu, const T_scale& beta) {
      static const std::string function("stan::prob::gumbel_log");

      using std::log;
      using std::exp;
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
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive(function, "Scale parameter", beta);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", beta);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> beta_vec(beta);
      size_t N = max_size(y, mu, beta);

      DoubleVectorView<true,is_vector<T_scale>::value> inv_beta(length(beta));
      DoubleVectorView<include_summand<propto,T_scale>::value,
                       is_vector<T_scale>::value> log_beta(length(beta));
      for (size_t i = 0; i < length(beta); i++) {
        inv_beta[i] = 1.0 / value_of(beta_vec[i]);
        if (include_summand<propto,T_scale>::value)
          log_beta[i] = log(value_of(beta_vec[i]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
      
        // reusable subexpression values
        const double y_minus_mu_over_beta 
          = (y_dbl - mu_dbl) * inv_beta[n];

        // log probability
        if (include_summand<propto,T_scale>::value)
          logp -= log_beta[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp += -y_minus_mu_over_beta - exp(-y_minus_mu_over_beta);

        // gradients
        double scaled_diff = inv_beta[n] * exp(-y_minus_mu_over_beta);
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= inv_beta[n] - scaled_diff;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += inv_beta[n] - scaled_diff;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += -inv_beta[n] + y_minus_mu_over_beta * inv_beta[n] 
            - scaled_diff * y_minus_mu_over_beta;
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_loc,T_scale>::type
    gumbel_log(const T_y& y, const T_loc& mu, const T_scale& beta) {
      return gumbel_log<false>(y,mu,beta);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    gumbel_cdf(const T_y& y, const T_loc& mu, const T_scale& beta) {
      static const std::string function("stan::prob::gumbel_cdf");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double cdf(1.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(beta)))
        return cdf;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", beta);
      check_positive(function, "Scale parameter", beta);
      check_consistent_sizes(function, 
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", beta);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> beta_vec(beta);
      size_t N = max_size(y, mu, beta);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / beta_dbl;
        const double rep_deriv = exp(-scaled_diff - exp(-scaled_diff)) 
          / beta_dbl;
        const double cdf_ = exp(-exp(-scaled_diff));
        cdf *= cdf_;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv / cdf_;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv / cdf_;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff / cdf_;
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
        for(size_t n = 0; n < stan::length(beta); ++n) 
          operands_and_partials.d_x3[n] *= cdf;
      }

      return operands_and_partials.to_var(cdf);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    gumbel_cdf_log(const T_y& y, const T_loc& mu, const T_scale& beta) {
      static const std::string function("stan::prob::gumbel_cdf_log");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double cdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(beta)))
        return cdf_log;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", beta);
      check_positive(function, "Scale parameter", beta);
      check_consistent_sizes(function, 
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", beta);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> beta_vec(beta);
      size_t N = max_size(y, mu, beta);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / beta_dbl;
        const double rep_deriv = exp(-scaled_diff) / beta_dbl;
        cdf_log -= exp(-scaled_diff);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] -= rep_deriv;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= rep_deriv * scaled_diff;
      }

      return operands_and_partials.to_var(cdf_log);
    }

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    gumbel_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& beta) {
      static const std::string function("stan::prob::gumbel_ccdf_log");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      double ccdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(beta)))
        return ccdf_log;

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Scale parameter", beta);
      check_positive(function, "Scale parameter", beta);
      check_consistent_sizes(function, 
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", beta);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> 
        operands_and_partials(y, mu, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> beta_vec(beta);
      size_t N = max_size(y, mu, beta);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
        const double scaled_diff = (y_dbl - mu_dbl) / beta_dbl;
        const double rep_deriv = exp(-scaled_diff - exp(-scaled_diff)) 
          / beta_dbl;
        const double ccdf_log_ = 1.0 - exp(-exp(-scaled_diff));
        ccdf_log += log(ccdf_log_);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv / ccdf_log_;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] += rep_deriv / ccdf_log_;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += rep_deriv * scaled_diff / ccdf_log_;
      }

      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    gumbel_rng(const double mu,
               const double beta,
               RNG& rng) {
      using boost::variate_generator;
      using boost::uniform_01;

      static const std::string function("stan::prob::gumbel_rng");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;


      check_finite(function, "Location parameter", mu);
      check_positive(function, "Scale parameter", beta); 

      variate_generator<RNG&, uniform_01<> >
        uniform01_rng(rng, uniform_01<>());
      return mu - beta * std::log(-std::log(uniform01_rng()));
    }
  }
}
#endif

