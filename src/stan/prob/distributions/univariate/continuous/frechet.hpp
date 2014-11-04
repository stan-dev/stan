#ifndef STAN__PROB__DISTRIBUTIONS__FRECHET_HPP
#define STAN__PROB__DISTRIBUTIONS__FRECHET_HPP

#include <boost/random/weibull_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // Frechet(y|alpha,sigma)     [y > 0;  alpha > 0;  sigma > 0]
    // FIXME: document
    template <bool propto,
              typename T_y, typename T_shape, typename T_scale>
    typename return_type<T_y,T_shape,T_scale>::type
    frechet_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      static const std::string function("stan::prob::frechet_log");

      using stan::error_handling::check_positive;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_positive_finite;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::multiply_log;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);
      check_positive(function, "Random variable", y);
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Shape parameter", alpha,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_shape,T_scale>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, alpha, sigma);

      DoubleVectorView<include_summand<propto,T_shape>::value,
        is_vector<T_shape>::value> log_alpha(length(alpha));
      for (size_t i = 0; i < length(alpha); i++)
        if (include_summand<propto,T_shape>::value)
          log_alpha[i] = log(value_of(alpha_vec[i]));
      
      DoubleVectorView<include_summand<propto,T_y,T_shape>::value,
        is_vector<T_y>::value> log_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y,T_shape>::value)
          log_y[i] = log(value_of(y_vec[i]));

      DoubleVectorView<include_summand<propto,T_shape,T_scale>::value,
        is_vector<T_scale>::value> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++)
        if (include_summand<propto,T_shape,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));

      DoubleVectorView<include_summand<propto,T_y,T_shape,T_scale>::value,
        is_vector<T_y>::value> inv_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y,T_shape,T_scale>::value)
          inv_y[i] = 1.0 / value_of(y_vec[i]);
      
      DoubleVectorView<include_summand<propto,T_y,T_shape,T_scale>::value,
        is_vector<T_y>::value | is_vector<T_shape>::value | is_vector<T_scale>::value>
        sigma_div_y_pow_alpha(N);
      for (size_t i = 0; i < N; i++)
        if (include_summand<propto,T_y,T_shape,T_scale>::value) {
          const double alpha_dbl = value_of(alpha_vec[i]);
          sigma_div_y_pow_alpha[i] = pow(value_of(inv_y[i]) * value_of(sigma_vec[i]), alpha_dbl);
        }

      agrad::OperandsAndPartials<T_y,T_shape,T_scale> operands_and_partials(y,alpha,sigma);
      for (size_t n = 0; n < N; n++) {
        const double alpha_dbl = value_of(alpha_vec[n]);
        if (include_summand<propto,T_shape>::value)
          logp += log_alpha[n];
        if (include_summand<propto,T_y,T_shape>::value)
          logp -= (alpha_dbl+1.0)*log_y[n];
        if (include_summand<propto,T_shape,T_scale>::value)
          logp += alpha_dbl*log_sigma[n];
        if (include_summand<propto,T_y,T_shape,T_scale>::value)
          logp -= sigma_div_y_pow_alpha[n];

        if (!is_constant_struct<T_y>::value) {
          const double inv_y_dbl = value_of(inv_y[n]);
          operands_and_partials.d_x1[n] 
            += -(alpha_dbl+1.0) * inv_y_dbl
            + alpha_dbl * sigma_div_y_pow_alpha[n] * inv_y_dbl;
        }
        if (!is_constant_struct<T_shape>::value) 
          operands_and_partials.d_x2[n] 
            += 1.0/alpha_dbl 
            + (1.0 - sigma_div_y_pow_alpha[n]) * (log_sigma[n] - log_y[n]);
        if (!is_constant_struct<T_scale>::value) 
          operands_and_partials.d_x3[n] 
            += alpha_dbl / value_of(sigma_vec[n]) * (1 - sigma_div_y_pow_alpha[n]);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_shape, typename T_scale>
    inline
    typename return_type<T_y,T_shape,T_scale>::type
    frechet_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {
      return frechet_log<false>(y,alpha,sigma);
    }

    template <typename T_y, typename T_shape, typename T_scale>
    typename return_type<T_y,T_shape,T_scale>::type
    frechet_cdf(const T_y& y, const T_shape& alpha, const T_scale& sigma) {

      static const std::string function("stan::prob::frechet_cdf");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_nonnegative;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(sigma)))
        return 1.0;

      double cdf(1.0);
      check_positive(function, "Random variable", y);
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", sigma);
      
      agrad::OperandsAndPartials<T_y, T_shape, T_scale> 
        operands_and_partials(y, alpha, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, sigma, alpha);
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double pow_ = pow(sigma_dbl / y_dbl, alpha_dbl);
        const double cdf_ = exp(-pow_);

        //cdf
        cdf *= cdf_;

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += pow_ * alpha_dbl / y_dbl;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n] += pow_ * log(y_dbl / sigma_dbl);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= pow_ * alpha_dbl / sigma_dbl;
      }

      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= cdf;
      if (!is_constant_struct<T_shape>::value)
        for (size_t n = 0; n < stan::length(alpha); ++n) 
          operands_and_partials.d_x2[n] *= cdf;
      if (!is_constant_struct<T_scale>::value)
        for (size_t n = 0; n < stan::length(sigma); ++n) 
          operands_and_partials.d_x3[n] *= cdf;

      return operands_and_partials.to_var(cdf);    
    }
   
    template <typename T_y, typename T_shape, typename T_scale>
    typename return_type<T_y,T_shape,T_scale>::type
    frechet_cdf_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {

      static const std::string function("stan::prob::frechet_cdf_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_nonnegative;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(sigma)))
        return 0.0;

      double cdf_log(0.0);
      check_positive(function, "Random variable", y);
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", sigma);
      
      agrad::OperandsAndPartials<T_y, T_shape, T_scale> 
        operands_and_partials(y, alpha, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, sigma, alpha);
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double pow_ = pow(sigma_dbl / y_dbl, alpha_dbl);

        //cdf_log
        cdf_log -= pow_;

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += pow_ * alpha_dbl / y_dbl;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n] += pow_ * log(y_dbl / sigma_dbl);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= pow_ * alpha_dbl / sigma_dbl;
      }

      return operands_and_partials.to_var(cdf_log);    
    }

    template <typename T_y, typename T_shape, typename T_scale>
    typename return_type<T_y,T_shape,T_scale>::type
    frechet_ccdf_log(const T_y& y, const T_shape& alpha, const T_scale& sigma) {

      static const std::string function("stan::prob::frechet_ccdf_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_nonnegative;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(sigma)))
        return 0.0;

      double ccdf_log(0.0);
      check_positive(function, "Random variable", y);
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", sigma);
      
      agrad::OperandsAndPartials<T_y, T_shape, T_scale> 
        operands_and_partials(y, alpha, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, sigma, alpha);
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double pow_ = pow(sigma_dbl / y_dbl, alpha_dbl);
		const double exp_ = exp(-pow_);

        //ccdf_log
        ccdf_log += log(1 - exp_);

        //gradients
        const double rep_deriv_ = pow_ / ( 1.0 / exp_ - 1 );
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= alpha_dbl / y_dbl * rep_deriv_;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n] -= log(y_dbl / sigma_dbl) * rep_deriv_;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += alpha_dbl / sigma_dbl * rep_deriv_;
      }

      return operands_and_partials.to_var(ccdf_log);    
    }

    template <class RNG>
    inline double
    frechet_rng(const double alpha,
                const double sigma,
                RNG& rng) {
      using boost::variate_generator;
      using boost::random::weibull_distribution;

      static const std::string function("stan::prob::frechet_rng");

      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_positive;
  
      check_finite(function, "Shape parameter", alpha);
      check_positive(function, "Shape parameter", alpha);
      check_not_nan(function, "Scale parameter", sigma);
      check_positive(function, "Scale parameter", sigma);

      variate_generator<RNG&, weibull_distribution<> >
        weibull_rng(rng, weibull_distribution<>(alpha, 1.0/sigma));
      return 1.0 / weibull_rng();
    }
  }
}
#endif
