#ifndef STAN_MATH_PRIM_SCAL_PROB_BERNOULLI_LOGIT_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_BERNOULLI_LOGIT_LOG_HPP

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
#include <cmath>

namespace stan {

  namespace math {

    // Bernoulli(n|inv_logit(theta))   [0 <= n <= 1;   -inf <= theta <= inf]
    // FIXME: documentation
    template <bool propto, typename T_n, typename T_prob>
    typename return_type<T_prob>::type
    bernoulli_logit_log(const T_n& n, const T_prob& theta) {
      static const char* function("stan::math::bernoulli_logit_log");
      typedef typename stan::partials_return_type<T_n, T_prob>::type
        T_partials_return;

      using stan::is_constant_struct;
      using stan::math::check_not_nan;
      using stan::math::check_bounded;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::math::include_summand;
      using stan::math::log1p;
      using stan::math::inv_logit;
      using std::exp;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(theta)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args (here done over var, which should be OK)
      check_bounded(function, "n", n, 0, 1);
      check_not_nan(function, "Logit transformed probability parameter", theta);
      check_consistent_sizes(function,
                             "Random variable", n,
                             "Probability parameter", theta);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_prob>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_prob> theta_vec(theta);
      size_t N = max_size(n, theta);
      OperandsAndPartials<T_prob> operands_and_partials(theta);

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const int n_int = value_of(n_vec[n]);
        const T_partials_return theta_dbl = value_of(theta_vec[n]);

        // reusable subexpression values
        const int sign = 2*n_int-1;
        const T_partials_return ntheta = sign * theta_dbl;
        const T_partials_return exp_m_ntheta = exp(-ntheta);

        // Handle extreme values gracefully using Taylor approximations.
        static const double cutoff = 20.0;
        if (ntheta > cutoff)
          logp -= exp_m_ntheta;
        else if (ntheta < -cutoff)
          logp += ntheta;
        else
          logp -= log1p(exp_m_ntheta);

        // gradients
        if (!is_constant_struct<T_prob>::value) {
          static const double cutoff = 20.0;
          if (ntheta > cutoff)
            operands_and_partials.d_x1[n] -= exp_m_ntheta;
          else if (ntheta < -cutoff)
            operands_and_partials.d_x1[n] += sign;
          else
            operands_and_partials.d_x1[n] += sign * exp_m_ntheta
              / (exp_m_ntheta + 1);
        }
      }
      return operands_and_partials.to_var(logp, theta);
    }

    template <typename T_n,
              typename T_prob>
    inline
    typename return_type<T_prob>::type
    bernoulli_logit_log(const T_n& n,
                        const T_prob& theta) {
      return bernoulli_logit_log<false>(n, theta);
    }

  }  // namespace math
}  // namespace stan
#endif
