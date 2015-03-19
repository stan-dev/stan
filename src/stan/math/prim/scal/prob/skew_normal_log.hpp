#ifndef STAN__MATH__PRIM__SCAL__PROB__SKEW_NORMAL_LOG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__SKEW_NORMAL_LOG_HPP

#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/fun/owens_t.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>

namespace stan {

  namespace prob {

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename return_type<T_y,T_loc,T_scale,T_shape>::type
    skew_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
                    const T_shape& alpha) {
      static const char* function("stan::prob::skew_normal_log");
      typedef typename stan::partials_return_type<T_y,T_loc,
                                                  T_scale,T_shape>::type
        T_partials_return;

      using std::log;
      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(y)
            && stan::length(mu)
            && stan::length(sigma)
            && stan::length(alpha)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_finite(function, "Shape parameter", alpha);
      check_positive(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Shape paramter", alpha);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale,T_shape>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_shape>
        operands_and_partials(y, mu, sigma, alpha);

      using boost::math::erfc;
      using boost::math::erf;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, mu, sigma, alpha);

      VectorBuilder<true, T_partials_return, T_scale> inv_sigma(length(sigma));
      VectorBuilder<include_summand<propto,T_scale>::value,
                    T_partials_return, T_scale> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
        if (include_summand<propto,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return alpha_dbl = value_of(alpha_vec[n]);

        // reusable subexpression values
        const T_partials_return y_minus_mu_over_sigma
          = (y_dbl - mu_dbl) * inv_sigma[n];
        const double pi_dbl = stan::math::pi();

        // log probability
        if (include_summand<propto>::value)
          logp -=  0.5 * log(2.0 * pi_dbl);
        if (include_summand<propto, T_scale>::value)
          logp -= log(sigma_dbl);
        if (include_summand<propto,T_y, T_loc, T_scale>::value)
          logp -= y_minus_mu_over_sigma * y_minus_mu_over_sigma / 2.0;
        if (include_summand<propto,T_y,T_loc,T_scale,T_shape>::value)
          logp += log(erfc(-alpha_dbl * y_minus_mu_over_sigma
                           / std::sqrt(2.0)));

        // gradients
        T_partials_return deriv_logerf
          = 2.0 / std::sqrt(pi_dbl)
          * exp(-alpha_dbl * y_minus_mu_over_sigma / std::sqrt(2.0)
                * alpha_dbl * y_minus_mu_over_sigma / std::sqrt(2.0))
          / (1 + erf(alpha_dbl * y_minus_mu_over_sigma
                     / std::sqrt(2.0)));
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n]
            += -y_minus_mu_over_sigma / sigma_dbl
            + deriv_logerf * alpha_dbl / (sigma_dbl * std::sqrt(2.0)) ;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n]
            += y_minus_mu_over_sigma / sigma_dbl
            + deriv_logerf * -alpha_dbl / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n]
            += -1.0 / sigma_dbl
            + y_minus_mu_over_sigma * y_minus_mu_over_sigma / sigma_dbl
            - deriv_logerf * y_minus_mu_over_sigma * alpha_dbl
            / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x4[n]
            += deriv_logerf * y_minus_mu_over_sigma / std::sqrt(2.0);
      }
      return operands_and_partials.to_var(logp,y,mu,sigma,alpha);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline
    typename return_type<T_y,T_loc,T_scale, T_shape>::type
    skew_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
                    const T_shape& alpha) {
      return skew_normal_log<false>(y,mu,sigma,alpha);
    }
  }
}
#endif

