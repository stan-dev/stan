#ifndef STAN_MATH_PRIM_SCAL_PROB_EXPONENTIAL_CDF_HPP
#define STAN_MATH_PRIM_SCAL_PROB_EXPONENTIAL_CDF_HPP

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
#include <cmath>

namespace stan {

  namespace math {

    /**
     * Calculates the exponential cumulative distribution function for
     * the given y and beta.
     *
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     *
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_inv_scale>
    typename return_type<T_y, T_inv_scale>::type
    exponential_cdf(const T_y& y, const T_inv_scale& beta) {
      typedef typename stan::partials_return_type<T_y, T_inv_scale>::type
        T_partials_return;

      static const char* function("stan::math::exponential_cdf");

      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      using stan::math::value_of;
      using std::exp;

      T_partials_return cdf(1.0);
      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(beta)))
        return cdf;

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
        const T_partials_return one_m_exp = 1.0 - exp(-beta_dbl * y_dbl);

        // cdf
        cdf *= one_m_exp;
      }

      for (size_t n = 0; n < N; n++) {
        const T_partials_return beta_dbl = value_of(beta_vec[n]);
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return one_m_exp = 1.0 - exp(-beta_dbl * y_dbl);

        // gradients
        T_partials_return rep_deriv = exp(-beta_dbl * y_dbl) / one_m_exp;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv * beta_dbl * cdf;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[n] += rep_deriv * y_dbl * cdf;
      }

      return operands_and_partials.to_var(cdf, y, beta);
    }
  }
}

#endif
