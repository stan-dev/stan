#ifndef STAN__PROB__DISTRIBUTIONS__LOMAX_HPP
#define STAN__PROB__DISTRIBUTIONS__LOMAX_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/agrad/partials_vari.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>


namespace stan {
  namespace prob {

    // lomax(y|lambda,alpha)  [y >= 0;  lambda > 0;  alpha > 0]
    template <bool propto,
              typename T_y, typename T_scale, typename T_shape>
    typename return_type<T_y,T_scale,T_shape>::type
    lomax_log(const T_y& y, const T_scale& lambda, const T_shape& alpha) {
      static const char* function = "stan::prob::lomax_log(%1%)";
      
      using stan::math::value_of;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(lambda) 
            && stan::length(alpha)))
        return 0.0;
      
      // set up return value accumulator
      double logp(0.0);
      
      // validate args (here done over var, which should be OK)
      check_nonnegative(function, y, "Random variable", &logp);
      check_not_nan(function, y, "Random variable", &logp);
      check_finite(function, lambda, "Scale parameter", &logp);
      check_positive(function, lambda, "Scale parameter", &logp);
      check_finite(function, alpha, "Shape parameter", &logp);
      check_positive(function, alpha, "Shape parameter", &logp);
      check_consistent_sizes(function,
                             y,lambda,alpha,
                             "Random variable","Scale parameter",
                             "Shape parameter",
                             &logp);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_scale,T_shape>::value)
        return 0.0;
      
      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> lambda_vec(lambda);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, lambda, alpha);

      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y,T_scale,T_shape> 
        operands_and_partials(y, lambda, alpha);
      
      DoubleVectorView<include_summand<propto,T_y,T_scale,T_shape>::value,
                       is_vector<T_y>::value> log1p_y_over_lambda(length(N));
      if (include_summand<propto,T_y,T_scale,T_shape>::value)
        for (size_t n = 0; n < N; n++)
          log1p_y_over_lambda[n] = log(1+value_of(y_vec[n]) 
                                       / value_of(lambda_vec[n]));

      DoubleVectorView<include_summand<propto,T_scale>::value,
                       is_vector<T_scale>::value> log_lambda(length(lambda));
      if (include_summand<propto,T_scale>::value)
        for (size_t n = 0; n < length(lambda); n++)
          log_lambda[n] = log(value_of(lambda_vec[n]));

      DoubleVectorView<include_summand<propto,T_shape>::value,
                       is_vector<T_shape>::value> log_alpha(length(alpha));
      if (include_summand<propto,T_shape>::value)
        for (size_t n = 0; n < length(alpha); n++)
          log_alpha[n] = log(value_of(alpha_vec[n]));

      DoubleVectorView<include_summand<propto,T_shape>::value,
                       is_vector<T_shape>::value> inv_alpha(length(alpha));
      if (include_summand<propto,T_shape>::value)
        for (size_t n = 0; n < length(alpha); n++)
          inv_alpha[n] = 1 / value_of(alpha_vec[n]);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);

        // log probability
        if (include_summand<propto,T_shape>::value)
          logp += log(alpha_dbl);
        if (include_summand<propto,T_scale>::value)
          logp -= log_lambda[n];
        if (include_summand<propto,T_y,T_scale,T_shape>::value)
          logp -= (alpha_dbl + 1) * log1p_y_over_lambda[n];
  
        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += (lambda_dbl - alpha_dbl) 
            / (lambda_dbl + y_dbl);
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] += ((alpha_dbl-2) * y_dbl - lambda_dbl) 
            / (lambda_dbl * (lambda_dbl + y_dbl));
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x3[n] += inv_alpha[n] - log1p_y_over_lambda[n];
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_scale, typename T_shape>
    inline
    typename return_type<T_y,T_scale,T_shape>::type
    lomax_log(const T_y& y, const T_scale& lambda, const T_shape& alpha) {
      return lomax_log<false>(y,lambda,alpha);
    }
  }
}
#endif
