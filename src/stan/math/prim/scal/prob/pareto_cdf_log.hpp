#ifndef STAN__MATH__PRIM__SCAL__PROB__PARETO_CDF_LOG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__PARETO_CDF_LOG_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>


namespace stan {
  namespace prob {

    template <typename T_y, typename T_scale, typename T_shape>
    typename return_type<T_y, T_scale, T_shape>::type
    pareto_cdf_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
      typedef typename stan::partials_return_type<T_y,T_scale,T_shape>::type
        T_partials_return;

      // Size checks
      if ( !( stan::length(y) && stan::length(y_min) && stan::length(alpha) ) )
        return 0.0;

      // Check errors
      static const char* function("stan::prob::pareto_cdf_log");

      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_greater_or_equal;
      using stan::math::check_consistent_sizes;
      using stan::math::check_nonnegative;
      using stan::math::value_of;

      T_partials_return P(0.0);

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Scale parameter", y_min);
      check_positive_finite(function, "Shape parameter", alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Scale parameter", y_min,
                             "Shape parameter", alpha);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> y_min_vec(y_min);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, y_min, alpha);

      agrad::OperandsAndPartials<T_y, T_scale, T_shape>
        operands_and_partials(y, y_min, alpha);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero

      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) < value_of(y_min_vec[i]))
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              y,y_min,alpha);
      }

      // Compute vectorized cdf_log and its gradients

      for (size_t n = 0; n < N; n++) {
          // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          return operands_and_partials.to_var(0.0,y,y_min,alpha);
        }
          // Pull out values
        const T_partials_return log_dbl = log( value_of(y_min_vec[n])
                                               / value_of(y_vec[n]) );
        const T_partials_return y_min_inv_dbl = 1.0 / value_of(y_min_vec[n]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
          // Compute
        const T_partials_return Pn = 1.0 - exp(alpha_dbl * log_dbl );
                P += log(Pn);
          if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n]
            += alpha_dbl * y_min_inv_dbl * exp((alpha_dbl + 1) * log_dbl) / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n]
            -= alpha_dbl * y_min_inv_dbl * exp( alpha_dbl * log_dbl ) / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x3[n]
            -= exp( alpha_dbl * log_dbl ) * log_dbl / Pn;
      }

      return operands_and_partials.to_var(P,y,y_min,alpha);
    }
  }
}
#endif
