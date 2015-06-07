#ifndef STAN_MATH_PRIM_SCAL_PROB_VON_MISES_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_VON_MISES_LOG_HPP

#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/fun/modified_bessel_first_kind.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <cmath>

namespace stan {

  namespace math {

    template<bool propto,
             typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa) {
      static char const* const function = "stan::math::von_mises_log";
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(kappa)))
        return 0.0;

      using stan::is_constant_struct;
      using stan::math::check_finite;
      using stan::math::check_positive_finite;
      using stan::math::check_greater;
      using stan::math::check_nonnegative;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      using stan::math::modified_bessel_first_kind;
      using std::log;

      // Result accumulator.
      T_partials_return logp = 0.0;

      // Validate arguments.
      check_finite(function, "Random variable", y);
      check_finite(function, "Location paramter", mu);
      check_positive_finite(function, "Scale parameter", kappa);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", kappa);


      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_loc, T_scale>::value)
        return logp;

      // Determine constants.
      const bool y_const = is_constant_struct<T_y>::value;
      const bool mu_const = is_constant_struct<T_loc>::value;
      const bool kappa_const = is_constant_struct<T_scale>::value;

      // Determine which expensive computations to perform.
      const bool compute_bessel0 = include_summand<propto, T_scale>::value;
      const bool compute_bessel1 = !kappa_const;
      const double TWO_PI = 2.0 * stan::math::pi();

      // Wrap scalars into vector views.
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> kappa_vec(kappa);

      VectorBuilder<true, T_partials_return, T_scale> kappa_dbl(length(kappa));
      VectorBuilder<include_summand<propto, T_scale>::value,
                    T_partials_return, T_scale> log_bessel0(length(kappa));
      for (size_t i = 0; i < length(kappa); i++) {
        kappa_dbl[i] = value_of(kappa_vec[i]);
        if (include_summand<propto, T_scale>::value)
          log_bessel0[i]
            = log(modified_bessel_first_kind(0, value_of(kappa_vec[i])));
      }

      OperandsAndPartials<T_y, T_loc, T_scale> oap(y, mu, kappa);

      size_t N = max_size(y, mu, kappa);

      for (size_t n = 0; n < N; n++) {
        // Extract argument values.
        const T_partials_return y_ = value_of(y_vec[n]);
        const T_partials_return y_dbl =  y_ - floor(y_ / TWO_PI) * TWO_PI;
        const T_partials_return mu_dbl = value_of(mu_vec[n]);

        // Reusable values.
        T_partials_return bessel0 = 0;
        if (compute_bessel0)
          bessel0 = modified_bessel_first_kind(0, kappa_dbl[n]);
        T_partials_return bessel1 = 0;
        if (compute_bessel1)
          bessel1 = modified_bessel_first_kind(-1, kappa_dbl[n]);
        const T_partials_return kappa_sin = kappa_dbl[n] * sin(mu_dbl - y_dbl);
        const T_partials_return kappa_cos = kappa_dbl[n] * cos(mu_dbl - y_dbl);

        // Log probability.
        if (include_summand<propto>::value)
          logp -= LOG_TWO_PI;
        if (include_summand<propto, T_scale>::value)
          logp -= log_bessel0[n];
        if (include_summand<propto, T_y, T_loc, T_scale>::value)
          logp += kappa_cos;

        // Gradient.
        if (!y_const)
          oap.d_x1[n] += kappa_sin;
        if (!mu_const)
          oap.d_x2[n] -= kappa_sin;
        if (!kappa_const)
          oap.d_x3[n] += kappa_cos / kappa_dbl[n] - bessel1 / bessel0;
      }

      return oap.to_var(logp, y, mu, kappa);
    }

    template<typename T_y, typename T_loc, typename T_scale>
    inline typename return_type<T_y, T_loc, T_scale>::type
    von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa) {
      return von_mises_log<false>(y, mu, kappa);
    }
  }
}
#endif
