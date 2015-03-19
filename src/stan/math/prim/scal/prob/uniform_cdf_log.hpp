#ifndef STAN__MATH__PRIM__SCAL__PROB__UNIFORM_CDF_LOG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__UNIFORM_CDF_LOG_HPP

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace prob {

    template <typename T_y, typename T_low, typename T_high>
    typename return_type<T_y,T_low,T_high>::type
    uniform_cdf_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      static const char* function("stan::prob::uniform_cdf_log");
      typedef typename stan::partials_return_type<T_y,T_low,T_high>::type
        T_partials_return;

      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_greater;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(alpha)
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      T_partials_return cdf_log(0.0);
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Lower bound parameter", alpha);
      check_finite(function, "Upper bound parameter", beta);
      check_greater(function, "Upper bound parameter", beta, alpha);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Lower bound parameter", alpha,
                             "Upper bound parameter", beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_low> alpha_vec(alpha);
      VectorView<const T_high> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);

      agrad::OperandsAndPartials<T_y,T_low,T_high>
        operands_and_partials(y,alpha,beta);

      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        if (y_dbl < value_of(alpha_vec[n])
            || y_dbl > value_of(beta_vec[n]))
          return stan::math::negative_infinity();
        if (y_dbl == value_of(beta_vec[n]))
          return operands_and_partials.to_var(0.0,y,alpha,beta);
      }

      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
        const T_partials_return beta_dbl = value_of(beta_vec[n]);
        const T_partials_return b_min_a = beta_dbl - alpha_dbl;
        const T_partials_return cdf_log_ = (y_dbl - alpha_dbl) / b_min_a;

        //cdf_log
        cdf_log += log(cdf_log_);

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += 1.0 / b_min_a / cdf_log_;
        if (!is_constant_struct<T_low>::value)
          operands_and_partials.d_x2[n] += (y_dbl - beta_dbl) / b_min_a
            / b_min_a / cdf_log_;
        if (!is_constant_struct<T_high>::value)
          operands_and_partials.d_x3[n] -= 1.0 / b_min_a;
      }

      return operands_and_partials.to_var(cdf_log,y,alpha,beta);
    }
  }
}
#endif
