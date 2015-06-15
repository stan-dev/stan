#ifndef STAN_MATH_PRIM_SCAL_PROB_BERNOULLI_CDF_HPP
#define STAN_MATH_PRIM_SCAL_PROB_BERNOULLI_CDF_HPP

#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {

    // Bernoulli CDF
    template <typename T_n, typename T_prob>
    typename return_type<T_prob>::type
    bernoulli_cdf(const T_n& n, const T_prob& theta) {
      static const char* function("stan::math::bernoulli_cdf");
      typedef typename stan::partials_return_type<T_n, T_prob>::type
        T_partials_return;

      using stan::math::check_finite;
      using stan::math::check_bounded;
      using stan::math::check_consistent_sizes;
      using stan::math::include_summand;

      // Ensure non-zero argument lenghts
      if (!(stan::length(n) && stan::length(theta)))
        return 1.0;

      T_partials_return P(1.0);

      // Validate arguments
      check_finite(function, "Probability parameter", theta);
      check_bounded(function, "Probability parameter", theta, 0.0, 1.0);
      check_consistent_sizes(function,
                             "Random variable", n,
                             "Probability parameter", theta);

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_prob> theta_vec(theta);
      size_t size = max_size(n, theta);

      // Compute vectorized CDF and gradient
      using stan::math::value_of;
      OperandsAndPartials<T_prob> operands_and_partials(theta);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0)
          return operands_and_partials.to_var(0.0, theta);
      }

      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= 1)
          continue;

        const T_partials_return Pi = 1 - value_of(theta_vec[i]);

        P *= Pi;

        if (!is_constant_struct<T_prob>::value)
          operands_and_partials.d_x1[i] += - 1 / Pi;
      }

      if (!is_constant_struct<T_prob>::value) {
        for (size_t i = 0; i < stan::length(theta); ++i)
          operands_and_partials.d_x1[i] *= P;
      }
      return operands_and_partials.to_var(P, theta);
    }

  }  // namespace math
}  // namespace stan
#endif
