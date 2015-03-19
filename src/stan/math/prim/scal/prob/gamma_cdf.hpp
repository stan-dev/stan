#ifndef STAN__MATH__PRIM__SCAL__PROB__GAMMA_CDF_HPP
#define STAN__MATH__PRIM__SCAL__PROB__GAMMA_CDF_HPP

#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/gamma_p.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

#include <stan/math/prim/scal/fun/grad_reg_inc_gamma.hpp>

namespace stan {

  namespace prob {

    /**
     * The cumulative density function for a gamma distribution for y
     * with the specified shape and inverse scale parameters.
     *
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_shape, typename T_inv_scale>
    typename return_type<T_y,T_shape,T_inv_scale>::type
    gamma_cdf(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      // Size checks
      if (!(stan::length(y) && stan::length(alpha) && stan::length(beta)))
        return 1.0;
      typedef typename stan::partials_return_type<T_y,T_shape,
                                                  T_inv_scale>::type
        T_partials_return;

      // Error checks
      static const char* function("stan::prob::gamma_cdf");

      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::check_greater_or_equal;
      using stan::math::check_less_or_equal;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using boost::math::tools::promote_args;

      T_partials_return P(1.0);

      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Scale parameter", beta);
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Shape parameter", alpha,
                             "Scale Parameter", beta);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);

      agrad::OperandsAndPartials<T_y, T_shape, T_inv_scale>
        operands_and_partials(y, alpha, beta);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero

      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0)
          return operands_and_partials.to_var(0.0,y,alpha,beta);
      }

      // Compute CDF and its gradients
      using stan::math::gamma_p;
      using stan::math::digamma;
      using boost::math::tgamma;
      using std::exp;
      using std::pow;

      // Cache a few expensive function calls if nu is a parameter
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape> gamma_vec(stan::length(alpha));
      VectorBuilder<!is_constant_struct<T_shape>::value,
                    T_partials_return, T_shape>
        digamma_vec(stan::length(alpha));

      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const T_partials_return alpha_dbl = value_of(alpha_vec[i]);
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
        const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
        const T_partials_return beta_dbl = value_of(beta_vec[n]);
          // Compute
        const T_partials_return Pn = gamma_p(alpha_dbl, beta_dbl * y_dbl);
          P *= Pn;
          if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += beta_dbl * exp(-beta_dbl * y_dbl)
            * pow(beta_dbl * y_dbl,alpha_dbl-1) / tgamma(alpha_dbl) / Pn;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n]
            -= stan::math::grad_reg_inc_gamma(alpha_dbl, beta_dbl
                                              * y_dbl, gamma_vec[n],
                                              digamma_vec[n]) / Pn;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x3[n] += y_dbl  * exp(-beta_dbl * y_dbl)
            * pow(beta_dbl * y_dbl,alpha_dbl-1) / tgamma(alpha_dbl) / Pn;
      }

      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < stan::length(y); ++n)
          operands_and_partials.d_x1[n] *= P;
      if (!is_constant_struct<T_shape>::value)
        for (size_t n = 0; n < stan::length(alpha); ++n)
          operands_and_partials.d_x2[n] *= P;
      if (!is_constant_struct<T_inv_scale>::value)
        for (size_t n = 0; n < stan::length(beta); ++n)
          operands_and_partials.d_x3[n] *= P;

      return operands_and_partials.to_var(P,y,alpha,beta);
    }
  }
}

#endif
