#ifndef STAN_MATH_PRIM_SCAL_PROB_LOGISTIC_CCDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_LOGISTIC_CCDF_LOG_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1p.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/contains_nonconstant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/prob/logistic_log.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <cmath>
#include <limits>

namespace stan {
  namespace math {

    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y, T_loc, T_scale>::type
    logistic_ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      typedef typename stan::partials_return_type<T_y, T_loc, T_scale>::type
        T_partials_return;

      // Size checks
      if ( !( stan::length(y) && stan::length(mu) && stan::length(sigma) ) )
        return 0.0;

      // Error checks
      static const char* function("stan::math::logistic_cdf_log");

      using stan::math::check_not_nan;
      using stan::math::check_positive_finite;
      using stan::math::check_finite;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using boost::math::tools::promote_args;
      using std::log;
      using std::exp;

      T_partials_return P(0.0);

      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);

      OperandsAndPartials<T_y, T_loc, T_scale>
        operands_and_partials(y, mu, sigma);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero

      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == -std::numeric_limits<double>::infinity())
          return operands_and_partials.to_var(0.0, y, mu, sigma);
      }

      // Compute vectorized cdf_log and its gradients
      for (size_t n = 0; n < N; n++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              y, mu, sigma);
        }

        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return mu_dbl = value_of(mu_vec[n]);
        const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
        const T_partials_return sigma_inv_vec = 1.0 / value_of(sigma_vec[n]);

        // Compute
        const T_partials_return Pn = 1.0 - 1.0 / (1.0 + exp(-(y_dbl - mu_dbl)
                                                            * sigma_inv_vec));
        P += log(Pn);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n]
            -= exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n]
            -= - exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] -= - (y_dbl - mu_dbl) * sigma_inv_vec
            * exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
      }

      return operands_and_partials.to_var(P, y, mu, sigma);
    }
  }
}
#endif
