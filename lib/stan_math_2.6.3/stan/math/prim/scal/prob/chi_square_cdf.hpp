#ifndef STAN_MATH_PRIM_SCAL_PROB_CHI_SQUARE_CDF_HPP
#define STAN_MATH_PRIM_SCAL_PROB_CHI_SQUARE_CDF_HPP

#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/gamma_p.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_gamma.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>
#include <limits>

namespace stan {

  namespace math {

    /**
     * Calculates the chi square cumulative distribution function for the given
     * variate and degrees of freedom.
     *
     * y A scalar variate.
     * nu Degrees of freedom.
     *
     * @return The cdf of the chi square distribution
     */
    template <typename T_y, typename T_dof>
    typename return_type<T_y, T_dof>::type
    chi_square_cdf(const T_y& y, const T_dof& nu) {
      static const char* function("stan::math::chi_square_cdf");
      typedef typename stan::partials_return_type<T_y, T_dof>::type
        T_partials_return;

      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      T_partials_return cdf(1.0);

      // Size checks
      if (!(stan::length(y) && stan::length(nu)))
        return cdf;

      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);

      OperandsAndPartials<T_y, T_dof>
        operands_and_partials(y, nu);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0)
          return operands_and_partials.to_var(0.0, y, nu);
      }

      // Compute CDF and its gradients
      using stan::math::gamma_p;
      using stan::math::digamma;
      using boost::math::tgamma;
      using std::exp;
      using std::pow;
      using std::exp;

      // Cache a few expensive function calls if nu is a parameter
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> gamma_vec(stan::length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> digamma_vec(stan::length(nu));

      if (!is_constant_struct<T_dof>::value) {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const T_partials_return alpha_dbl = value_of(nu_vec[i]) * 0.5;
          gamma_vec[i] = tgamma(alpha_dbl);
          digamma_vec[i] = digamma(alpha_dbl);
        }
      }

      // Compute vectorized CDF and gradient
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity())
          continue;

        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return alpha_dbl = value_of(nu_vec[n]) * 0.5;
        const T_partials_return beta_dbl = 0.5;

        // Compute
        const T_partials_return Pn = gamma_p(alpha_dbl, beta_dbl * y_dbl);

        cdf *= Pn;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += beta_dbl * exp(-beta_dbl * y_dbl)
            * pow(beta_dbl * y_dbl, alpha_dbl-1) / tgamma(alpha_dbl) / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n]
            -= 0.5 * stan::math::grad_reg_inc_gamma(alpha_dbl, beta_dbl
                                                    * y_dbl, gamma_vec[n],
                                                    digamma_vec[n]) / Pn;
      }

      if (!is_constant_struct<T_y>::value) {
        for (size_t n = 0; n < stan::length(y); ++n)
          operands_and_partials.d_x1[n] *= cdf;
      }
      if (!is_constant_struct<T_dof>::value) {
        for (size_t n = 0; n < stan::length(nu); ++n)
          operands_and_partials.d_x2[n] *= cdf;
      }

      return operands_and_partials.to_var(cdf, y, nu);
    }
  }
}
#endif
