#ifndef STAN_MATH_PRIM_SCAL_PROB_SCALED_INV_CHI_SQUARE_CDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_SCALED_INV_CHI_SQUARE_CDF_LOG_HPP

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_gamma.hpp>
#include <limits>
#include <cmath>


namespace stan {

  namespace math {

    template <typename T_y, typename T_dof, typename T_scale>
    typename return_type<T_y, T_dof, T_scale>::type
    scaled_inv_chi_square_cdf_log(const T_y& y, const T_dof& nu,
                                  const T_scale& s) {
      typedef typename stan::partials_return_type<T_y, T_dof, T_scale>::type
        T_partials_return;

      // Size checks
      if (!(stan::length(y) && stan::length(nu) && stan::length(s)))
        return 0.0;

      static const char* function("stan::math::scaled_inv_chi_square_cdf_log");

      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using std::exp;

      T_partials_return P(0.0);

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_positive_finite(function, "Scale parameter", s);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu,
                             "Scale parameter", s);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_scale> s_vec(s);
      size_t N = max_size(y, nu, s);

      OperandsAndPartials<T_y, T_dof, T_scale>
        operands_and_partials(y, nu, s);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0)
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              y, nu, s);
      }

      // Compute cdf_log and its gradients
      using stan::math::gamma_q;
      using stan::math::digamma;
      using boost::math::tgamma;
      using std::exp;
      using std::pow;
      using std::log;

      // Cache a few expensive function calls if nu is a parameter
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> gamma_vec(stan::length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> digamma_vec(stan::length(nu));

      if (!is_constant_struct<T_dof>::value) {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const T_partials_return half_nu_dbl = 0.5 * value_of(nu_vec[i]);
          gamma_vec[i] = tgamma(half_nu_dbl);
          digamma_vec[i] = digamma(half_nu_dbl);
        }
      }

      // Compute vectorized cdf_log and gradient
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }

        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return y_inv_dbl = 1.0 / y_dbl;
        const T_partials_return half_nu_dbl = 0.5 * value_of(nu_vec[n]);
        const T_partials_return s_dbl = value_of(s_vec[n]);
        const T_partials_return half_s2_overx_dbl = 0.5 * s_dbl * s_dbl
          * y_inv_dbl;
        const T_partials_return half_nu_s2_overx_dbl
          = 2.0 * half_nu_dbl * half_s2_overx_dbl;

        // Compute
        const T_partials_return Pn = gamma_q(half_nu_dbl, half_nu_s2_overx_dbl);
        const T_partials_return gamma_p_deriv = exp(-half_nu_s2_overx_dbl)
          * pow(half_nu_s2_overx_dbl, half_nu_dbl-1) / tgamma(half_nu_dbl);

        P += log(Pn);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += half_nu_s2_overx_dbl * y_inv_dbl
            * gamma_p_deriv / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n]
            += (0.5 * stan::math::grad_reg_inc_gamma(half_nu_dbl,
                                                     half_nu_s2_overx_dbl,
                                                     gamma_vec[n],
                                                     digamma_vec[n])
                - half_s2_overx_dbl * gamma_p_deriv)
            / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += - 2.0 * half_nu_dbl * s_dbl
            * y_inv_dbl * gamma_p_deriv / Pn;
      }

      return operands_and_partials.to_var(P, y, nu, s);
    }
  }
}
#endif
