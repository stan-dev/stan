#ifndef STAN__MATH__PRIM__SCAL__PROB__INV_CHI_SQUARE_LOG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__INV_CHI_SQUARE_LOG_HPP

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_gamma.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of an inverse chi-squared density for y with the specified
     * degrees of freedom parameter.
     * The degrees of freedom prarameter must be greater than 0.
     * y must be greater than 0.
     *
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Inv-}}\chi^2_\nu \\
     \log (p (y \,|\, \nu)) &=& \log \left( \frac{2^{-\nu / 2}}{\Gamma (\nu / 2)} y^{- (\nu / 2 + 1)} \exp^{-1 / (2y)} \right) \\
     &=& - \frac{\nu}{2} \log(2) - \log (\Gamma (\nu / 2)) - (\frac{\nu}{2} + 1) \log(y) - \frac{1}{2y} \\
     & & \mathrm{ where } \; y > 0
     \f}
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <bool propto,
              typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    inv_chi_square_log(const T_y& y, const T_dof& nu) {
      static const char* function("stan::prob::inv_chi_square_log");
      typedef typename stan::partials_return_type<T_y,T_dof>::type
        T_partials_return;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(nu)))
        return 0.0;

      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      T_partials_return logp(0.0);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_not_nan(function, "Random variable", y);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu);


      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);

      for (size_t n = 0; n < length(y); n++)
        if (value_of(y_vec[n]) <= 0)
          return LOG_ZERO;

      using boost::math::digamma;
      using boost::math::lgamma;
      using stan::math::multiply_log;

      VectorBuilder<include_summand<propto,T_y,T_dof>::value,
                    T_partials_return, T_y> log_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y,T_dof>::value)
          log_y[i] = log(value_of(y_vec[i]));

      VectorBuilder<include_summand<propto,T_y>::value,
                    T_partials_return, T_y> inv_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y>::value)
          inv_y[i] = 1.0 / value_of(y_vec[i]);

      VectorBuilder<include_summand<propto,T_dof>::value,
                    T_partials_return, T_dof> lgamma_half_nu(length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof>
        digamma_half_nu_over_two(length(nu));
      for (size_t i = 0; i < length(nu); i++) {
        T_partials_return half_nu = 0.5 * value_of(nu_vec[i]);
        if (include_summand<propto,T_dof>::value)
          lgamma_half_nu[i] = lgamma(half_nu);
        if (!is_constant_struct<T_dof>::value)
          digamma_half_nu_over_two[i] = digamma(half_nu) * 0.5;
      }

      agrad::OperandsAndPartials<T_y, T_dof> operands_and_partials(y, nu);
      for (size_t n = 0; n < N; n++) {
        const T_partials_return nu_dbl = value_of(nu_vec[n]);
        const T_partials_return half_nu = 0.5 * nu_dbl;

        if (include_summand<propto,T_dof>::value)
          logp += nu_dbl * NEG_LOG_TWO_OVER_TWO - lgamma_half_nu[n];
        if (include_summand<propto,T_y,T_dof>::value)
          logp -= (half_nu+1.0) * log_y[n];
        if (include_summand<propto,T_y>::value)
          logp -= 0.5 * inv_y[n];

        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n]
            += -(half_nu+1.0) * inv_y[n] + 0.5 * inv_y[n] * inv_y[n];
        }
        if (!is_constant_struct<T_dof>::value) {
          operands_and_partials.d_x2[n]
            += NEG_LOG_TWO_OVER_TWO - digamma_half_nu_over_two[n]
            - 0.5*log_y[n];
        }
      }
      return operands_and_partials.to_var(logp,y,nu);
    }

    template <typename T_y, typename T_dof>
    inline
    typename return_type<T_y,T_dof>::type
    inv_chi_square_log(const T_y& y, const T_dof& nu) {
      return inv_chi_square_log<false>(y,nu);
    }
  }
}
#endif
