#ifndef STAN_MATH_PRIM_SCAL_PROB_BINOMIAL_CCDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_BINOMIAL_CCDF_LOG_HPP

#include <boost/random/binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/log_inv_logit.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <cmath>

namespace stan {

  namespace math {

    template <typename T_n, typename T_N, typename T_prob>
    typename return_type<T_prob>::type
    binomial_ccdf_log(const T_n& n, const T_N& N, const T_prob& theta) {
      static const char* function("stan::math::binomial_ccdf_log");
      typedef typename stan::partials_return_type<T_n, T_N, T_prob>::type
        T_partials_return;

      using stan::math::check_finite;
      using stan::math::check_bounded;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::math::include_summand;

      // Ensure non-zero arguments lenghts
      if (!(stan::length(n) && stan::length(N) && stan::length(theta)))
        return 0.0;

      T_partials_return P(0.0);

      // Validate arguments
      check_nonnegative(function, "Population size parameter", N);
      check_finite(function, "Probability parameter", theta);
      check_bounded(function, "Probability parameter", theta, 0.0, 1.0);
      check_consistent_sizes(function,
                             "Successes variable", n,
                             "Population size parameter", N,
                             "Probability parameter", theta);

      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_prob> theta_vec(theta);
      size_t size = max_size(n, N, theta);

      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      using stan::math::inc_beta;
      using stan::math::lbeta;
      using std::exp;
      using std::pow;
      using std::log;
      using std::exp;

      OperandsAndPartials<T_prob> operands_and_partials(theta);

      // Explicit return for extreme values
      // The gradients are technically ill-defined,
      // but treated as negative infinity
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0)
          return operands_and_partials.to_var(0.0, theta);
      }

      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= value_of(N_vec[i])) {
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              theta);
        }
        const T_partials_return n_dbl = value_of(n_vec[i]);
        const T_partials_return N_dbl = value_of(N_vec[i]);
        const T_partials_return theta_dbl = value_of(theta_vec[i]);
        const T_partials_return betafunc = exp(lbeta(N_dbl-n_dbl, n_dbl+1));
        const T_partials_return Pi = 1.0 - inc_beta(N_dbl - n_dbl, n_dbl + 1,
                                                    1 - theta_dbl);

        P += log(Pi);

        if (!is_constant_struct<T_prob>::value)
          operands_and_partials.d_x1[i] += pow(theta_dbl, n_dbl)
            * pow(1-theta_dbl, N_dbl-n_dbl-1) / betafunc / Pi;
      }

      return operands_and_partials.to_var(P, theta);
    }
  }
}
#endif
