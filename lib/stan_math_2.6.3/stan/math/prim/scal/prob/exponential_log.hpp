#ifndef STAN_MATH_PRIM_SCAL_PROB_EXPONENTIAL_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_EXPONENTIAL_LOG_HPP

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
     * The log of an exponential density for y with the specified
     * inverse scale parameter.
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     *
     \f{eqnarray*}{
     y
     &\sim&
     \mbox{\sf{Expon}}(\beta) \\
     \log (p (y \, |\, \beta) )
     &=&
     \log \left( \beta \exp^{-\beta y} \right) \\
     &=&
     \log (\beta) - \beta y \\
     & &
     \mathrm{where} \; y > 0
     \f}
     *
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <bool propto, typename T_y, typename T_inv_scale>
    typename return_type<T_y, T_inv_scale>::type
    exponential_log(const T_y& y, const T_inv_scale& beta) {
      static const char* function("stan::math::exponential_log");
      typedef typename stan::partials_return_type<T_y, T_inv_scale>::type
        T_partials_return;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(beta)))
        return 0.0;

      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using std::log;

      T_partials_return logp(0.0);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Inverse scale parameter", beta);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Inverse scale parameter", beta);


      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, beta);

      VectorBuilder<include_summand<propto, T_inv_scale>::value,
                    T_partials_return, T_inv_scale> log_beta(length(beta));
      for (size_t i = 0; i < length(beta); i++)
        if (include_summand<propto, T_inv_scale>::value)
          log_beta[i] = log(value_of(beta_vec[i]));

      OperandsAndPartials<T_y, T_inv_scale>
        operands_and_partials(y, beta);

      for (size_t n = 0; n < N; n++) {
        const T_partials_return beta_dbl = value_of(beta_vec[n]);
        const T_partials_return y_dbl = value_of(y_vec[n]);
        if (include_summand<propto, T_inv_scale>::value)
          logp += log_beta[n];
        if (include_summand<propto, T_y, T_inv_scale>::value)
          logp -= beta_dbl * y_dbl;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= beta_dbl;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[n] += 1 / beta_dbl - y_dbl;
      }
      return operands_and_partials.to_var(logp, y, beta);
    }

    template <typename T_y, typename T_inv_scale>
    inline
    typename return_type<T_y, T_inv_scale>::type
    exponential_log(const T_y& y, const T_inv_scale& beta) {
      return exponential_log<false>(y, beta);
    }
  }
}

#endif
