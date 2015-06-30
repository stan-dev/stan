#ifndef STAN_MATH_PRIM_SCAL_PROB_EXPONENTIAL_CCDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_EXPONENTIAL_CCDF_LOG_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {

    template <typename T_y, typename T_inv_scale>
    typename return_type<T_y, T_inv_scale>::type
    exponential_ccdf_log(const T_y& y, const T_inv_scale& beta) {
      typedef typename stan::partials_return_type<T_y, T_inv_scale>::type
        T_partials_return;

      static const char* function("stan::math::exponential_ccdf_log");

      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      T_partials_return ccdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(beta)))
        return ccdf_log;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Inverse scale parameter", beta);

      OperandsAndPartials<T_y, T_inv_scale>
        operands_and_partials(y, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, beta);
      for (size_t n = 0; n < N; n++) {
        const T_partials_return beta_dbl = value_of(beta_vec[n]);
        const T_partials_return y_dbl = value_of(y_vec[n]);
        // log ccdf
        ccdf_log += -beta_dbl * y_dbl;

        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= beta_dbl;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[n] -= y_dbl;
      }
      return operands_and_partials.to_var(ccdf_log, y, beta);
    }
  }
}

#endif
