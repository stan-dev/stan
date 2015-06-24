#ifndef STAN_MATH_PRIM_SCAL_PROB_RAYLEIGH_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_RAYLEIGH_LOG_HPP

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <cmath>

namespace stan {

  namespace math {

    template <bool propto,
              typename T_y, typename T_scale>
    typename return_type<T_y, T_scale>::type
    rayleigh_log(const T_y& y, const T_scale& sigma) {
      static const char* function("stan::math::rayleigh_log");
      typedef typename stan::partials_return_type<T_y, T_scale>::type
        T_partials_return;

      using std::log;
      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::math::include_summand;
      using std::log;

      // check if any vectors are zero length
      if (!(stan::length(y) && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_positive(function, "Scale parameter", sigma);
      check_positive(function, "Random variable", y);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      OperandsAndPartials<T_y, T_scale> operands_and_partials(y, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, sigma);

      VectorBuilder<true, T_partials_return, T_scale> inv_sigma(length(sigma));
      VectorBuilder<include_summand<propto, T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
        if (include_summand<propto, T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const T_partials_return y_dbl = value_of(y_vec[n]);

        // reusable subexpression values
        const T_partials_return y_over_sigma = y_dbl * inv_sigma[n];

        static double NEGATIVE_HALF = -0.5;

        // log probability
        if (include_summand<propto, T_scale>::value)
          logp -= 2.0 * log_sigma[n];
        if (include_summand<propto, T_y>::value)
          logp += log(y_dbl);
        // if (include_summand<propto, T_y, T_scale>::value)
        logp += NEGATIVE_HALF * y_over_sigma * y_over_sigma;

        // gradients
        T_partials_return scaled_diff = inv_sigma[n] * y_over_sigma;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += 1.0 / y_dbl - scaled_diff;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n]
            += y_over_sigma * scaled_diff - 2.0 * inv_sigma[n];
      }
      return operands_and_partials.to_var(logp, y, sigma);
    }

    template <typename T_y, typename T_scale>
    inline
    typename return_type<T_y, T_scale>::type
    rayleigh_log(const T_y& y, const T_scale& sigma) {
      return rayleigh_log<false>(y, sigma);
    }
  }
}
#endif
