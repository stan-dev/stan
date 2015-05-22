#ifndef STAN_MATH_PRIM_SCAL_PROB_SCALED_INV_CHI_SQUARE_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_SCALED_INV_CHI_SQUARE_LOG_HPP

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
#include <cmath>


namespace stan {

  namespace math {

    /**
     * The log of a scaled inverse chi-squared density for y with the
     * specified degrees of freedom parameter and scale parameter.
     *
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Inv-}}\chi^2(\nu, s^2) \\
     \log (p (y \, |\, \nu, s)) &=& \log \left( \frac{(\nu / 2)^{\nu / 2}}{\Gamma (\nu / 2)} s^\nu y^{- (\nu / 2 + 1)} \exp^{-\nu s^2 / (2y)} \right) \\
     &=& \frac{\nu}{2} \log(\frac{\nu}{2}) - \log (\Gamma (\nu / 2)) + \nu \log(s) - (\frac{\nu}{2} + 1) \log(y) - \frac{\nu s^2}{2y} \\
     & & \mathrm{ where } \; y > 0
     \f}
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param s Scale parameter.
     * @throw std::domain_error if nu is not greater than 0
     * @throw std::domain_error if s is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <bool propto,
              typename T_y, typename T_dof, typename T_scale>
    typename return_type<T_y, T_dof, T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s) {
      static const char* function("stan::math::scaled_inv_chi_square_log");
      typedef typename stan::partials_return_type<T_y, T_dof, T_scale>::type
        T_partials_return;

      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(nu)
            && stan::length(s)))
        return 0.0;

      T_partials_return logp(0.0);
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_positive_finite(function, "Scale parameter", s);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu,
                             "Scale parameter", s);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_dof, T_scale>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_scale> s_vec(s);
      size_t N = max_size(y, nu, s);

      for (size_t n = 0; n < N; n++) {
        if (value_of(y_vec[n]) <= 0)
          return LOG_ZERO;
      }

      using stan::math::lgamma;
      using stan::math::digamma;
      using stan::math::square;
      using std::log;

      VectorBuilder<include_summand<propto, T_dof, T_y, T_scale>::value,
                    T_partials_return, T_dof> half_nu(length(nu));
      for (size_t i = 0; i < length(nu); i++)
        if (include_summand<propto, T_dof, T_y, T_scale>::value)
          half_nu[i] = 0.5 * value_of(nu_vec[i]);

      VectorBuilder<include_summand<propto, T_dof, T_y>::value,
                    T_partials_return, T_y> log_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto, T_dof, T_y>::value)
          log_y[i] = log(value_of(y_vec[i]));

      VectorBuilder<include_summand<propto, T_dof, T_y, T_scale>::value,
                    T_partials_return, T_y> inv_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto, T_dof, T_y, T_scale>::value)
          inv_y[i] = 1.0 / value_of(y_vec[i]);

      VectorBuilder<include_summand<propto, T_dof, T_scale>::value,
                    T_partials_return, T_scale> log_s(length(s));
      for (size_t i = 0; i < length(s); i++)
        if (include_summand<propto, T_dof, T_scale>::value)
          log_s[i] = log(value_of(s_vec[i]));

      VectorBuilder<include_summand<propto, T_dof>::value,
                    T_partials_return, T_dof> log_half_nu(length(nu));
      VectorBuilder<include_summand<propto, T_dof>::value,
                    T_partials_return, T_dof> lgamma_half_nu(length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof>
        digamma_half_nu_over_two(length(nu));
      for (size_t i = 0; i < length(nu); i++) {
        if (include_summand<propto, T_dof>::value)
          lgamma_half_nu[i] = lgamma(half_nu[i]);
        if (include_summand<propto, T_dof>::value)
          log_half_nu[i] = log(half_nu[i]);
        if (!is_constant_struct<T_dof>::value)
          digamma_half_nu_over_two[i] = digamma(half_nu[i]) * 0.5;
      }

      OperandsAndPartials<T_y, T_dof, T_scale>
        operands_and_partials(y, nu, s);
      for (size_t n = 0; n < N; n++) {
        const T_partials_return s_dbl = value_of(s_vec[n]);
        const T_partials_return nu_dbl = value_of(nu_vec[n]);
        if (include_summand<propto, T_dof>::value)
          logp += half_nu[n] * log_half_nu[n] - lgamma_half_nu[n];
        if (include_summand<propto, T_dof, T_scale>::value)
          logp += nu_dbl * log_s[n];
        if (include_summand<propto, T_dof, T_y>::value)
          logp -= (half_nu[n]+1.0) * log_y[n];
        if (include_summand<propto, T_dof, T_y, T_scale>::value)
          logp -= half_nu[n] * s_dbl*s_dbl * inv_y[n];

        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n]
            += -(half_nu[n] + 1.0) * inv_y[n]
            + half_nu[n] * s_dbl*s_dbl * inv_y[n]*inv_y[n];
        }
        if (!is_constant_struct<T_dof>::value) {
          operands_and_partials.d_x2[n]
            += 0.5 * log_half_nu[n] + 0.5
            - digamma_half_nu_over_two[n]
            + log_s[n]
            - 0.5 * log_y[n]
            - 0.5* s_dbl*s_dbl * inv_y[n];
        }
        if (!is_constant_struct<T_scale>::value) {
          operands_and_partials.d_x3[n]
            += nu_dbl / s_dbl - nu_dbl * inv_y[n] * s_dbl;
        }
      }
      return operands_and_partials.to_var(logp, y, nu, s);
    }

    template <typename T_y, typename T_dof, typename T_scale>
    inline
    typename return_type<T_y, T_dof, T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s) {
      return scaled_inv_chi_square_log<false>(y, nu, s);
    }
  }
}
#endif
