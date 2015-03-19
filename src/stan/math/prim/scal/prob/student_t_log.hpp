#ifndef STAN__MATH__PRIM__SCAL__PROB__STUDENT_T_LOG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__STUDENT_T_LOG_HPP

#include <boost/random/student_t_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the Student-t density for the given y, nu, mean, and
     * scale parameter.  The scale parameter must be greater
     * than 0.
     *
     * \f{eqnarray*}{
     y &\sim& t_{\nu} (\mu, \sigma^2) \\
     \log (p (y \,|\, \nu, \mu, \sigma) ) &=& \log \left( \frac{\Gamma((\nu + 1) /2)}
     {\Gamma(\nu/2)\sqrt{\nu \pi} \sigma} \left( 1 + \frac{1}{\nu} (\frac{y - \mu}{\sigma})^2 \right)^{-(\nu + 1)/2} \right) \\
     &=& \log( \Gamma( (\nu+1)/2 )) - \log (\Gamma (\nu/2) - \frac{1}{2} \log(\nu \pi) - \log(\sigma)
     -\frac{\nu + 1}{2} \log (1 + \frac{1}{\nu} (\frac{y - \mu}{\sigma})^2)
     \f}
     *
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param mu The mean of the Student-t distribution.
     * @param sigma The scale parameter of the Student-t distribution.
     * @return The log of the Student-t density at y.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::domain_error if nu is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */
    template <bool propto, typename T_y, typename T_dof,
              typename T_loc, typename T_scale>
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu,
                  const T_scale& sigma) {
      static const char* function("stan::prob::student_t_log");
      typedef typename stan::partials_return_type<T_y,T_dof,T_loc,
                                                  T_scale>::type
        T_partials_return;

      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(nu)
            && stan::length(mu)
            && stan::length(sigma)))
        return 0.0;

      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, nu, mu, sigma);

      using std::log;
      using stan::math::digamma;
      using stan::math::lgamma;
      using stan::math::square;
      using stan::math::value_of;

      VectorBuilder<include_summand<propto,T_y,T_dof,T_loc,T_scale>::value,
                    T_partials_return, T_dof> half_nu(length(nu));
      for (size_t i = 0; i < length(nu); i++)
        if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
          half_nu[i] = 0.5 * value_of(nu_vec[i]);

      VectorBuilder<include_summand<propto,T_dof>::value,
                    T_partials_return, T_dof> lgamma_half_nu(length(nu));
      VectorBuilder<include_summand<propto,T_dof>::value,
                    T_partials_return, T_dof>
        lgamma_half_nu_plus_half(length(nu));
      if (include_summand<propto,T_dof>::value)
        for (size_t i = 0; i < length(nu); i++) {
          lgamma_half_nu[i] = lgamma(half_nu[i]);
          lgamma_half_nu_plus_half[i] = lgamma(half_nu[i] + 0.5);
        }

      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> digamma_half_nu(length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof>
        digamma_half_nu_plus_half(length(nu));
      if (!is_constant_struct<T_dof>::value)
        for (size_t i = 0; i < length(nu); i++) {
          digamma_half_nu[i] = digamma(half_nu[i]);
          digamma_half_nu_plus_half[i] = digamma(half_nu[i] + 0.5);
        }

      VectorBuilder<include_summand<propto,T_dof>::value,
                    T_partials_return, T_dof> log_nu(length(nu));
      for (size_t i = 0; i < length(nu); i++)
        if (include_summand<propto,T_dof>::value)
          log_nu[i] = log(value_of(nu_vec[i]));

      VectorBuilder<include_summand<propto,T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++)
        if (include_summand<propto,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));

      VectorBuilder<include_summand<propto,T_y,T_dof,T_loc,T_scale>::value,
                    T_partials_return, T_y, T_dof, T_loc, T_scale>
        square_y_minus_mu_over_sigma__over_nu(N);

      VectorBuilder<include_summand<propto,T_y,T_dof,T_loc,T_scale>::value,
                    T_partials_return, T_y, T_dof, T_loc, T_scale>
        log1p_exp(N);

      for (size_t i = 0; i < N; i++)
        if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value) {
          const T_partials_return y_dbl = value_of(y_vec[i]);
          const T_partials_return mu_dbl = value_of(mu_vec[i]);
          const T_partials_return sigma_dbl = value_of(sigma_vec[i]);
          const T_partials_return nu_dbl = value_of(nu_vec[i]);
          square_y_minus_mu_over_sigma__over_nu[i]
            = square((y_dbl - mu_dbl) / sigma_dbl) / nu_dbl;
          log1p_exp[i] = log1p(square_y_minus_mu_over_sigma__over_nu[i]);
        }

      agrad::OperandsAndPartials<T_y,T_dof,T_loc,T_scale>
        operands_and_partials(y,nu,mu,sigma);
      for (size_t n = 0; n < N; n++) {
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return nu_dbl = value_of(nu_vec[n]);
        if (include_summand<propto>::value)
          logp += NEG_LOG_SQRT_PI;
        if (include_summand<propto,T_dof>::value)
          logp += lgamma_half_nu_plus_half[n] - lgamma_half_nu[n]
            -  0.5 * log_nu[n];
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
          logp -= (half_nu[n] + 0.5)
            * log1p_exp[n];

        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n]
            += -(half_nu[n]+0.5)
            * 1.0 / (1.0 + square_y_minus_mu_over_sigma__over_nu[n])
            * (2.0 * (y_dbl - mu_dbl) / square(sigma_dbl) / nu_dbl);
        }
        if (!is_constant_struct<T_dof>::value) {
          const T_partials_return inv_nu = 1.0 / nu_dbl;
          operands_and_partials.d_x2[n]
            += 0.5*digamma_half_nu_plus_half[n] - 0.5*digamma_half_nu[n]
            - 0.5 * inv_nu
            - 0.5*log1p_exp[n]
            + (half_nu[n] + 0.5)
            * (1.0/(1.0 + square_y_minus_mu_over_sigma__over_nu[n])
               * square_y_minus_mu_over_sigma__over_nu[n] * inv_nu);
        }
        if (!is_constant_struct<T_loc>::value) {
          operands_and_partials.d_x3[n]
            -= (half_nu[n] + 0.5)
            / (1.0 + square_y_minus_mu_over_sigma__over_nu[n])
            * (2.0 * (mu_dbl - y_dbl) / (sigma_dbl*sigma_dbl*nu_dbl));
        }
        if (!is_constant_struct<T_scale>::value) {
          const T_partials_return inv_sigma = 1.0 / sigma_dbl;
          operands_and_partials.d_x4[n]
            += -inv_sigma
            + (nu_dbl + 1.0) / (1.0 + square_y_minus_mu_over_sigma__over_nu[n])
            * (square_y_minus_mu_over_sigma__over_nu[n] * inv_sigma);
        }
      }
      return operands_and_partials.to_var(logp,y,nu,mu,sigma);
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu,
                  const T_scale& sigma) {
      return student_t_log<false>(y,nu,mu,sigma);
    }
  }
}
#endif
