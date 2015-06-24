#ifndef STAN_MATH_PRIM_SCAL_PROB_POISSON_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_POISSON_LOG_HPP

#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <limits>

namespace stan {

  namespace math {

    // Poisson(n|lambda)  [lambda > 0;  n >= 0]
    template <bool propto, typename T_n, typename T_rate>
    typename return_type<T_rate>::type
    poisson_log(const T_n& n, const T_rate& lambda) {
      typedef typename stan::partials_return_type<T_n, T_rate>::type
        T_partials_return;

      static const char* function("stan::math::poisson_log");

      using boost::math::lgamma;
      using stan::math::check_consistent_sizes;
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::include_summand;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(lambda)))
        return 0.0;

      // set up return value accumulator
      T_partials_return logp(0.0);

      // validate args
      check_nonnegative(function, "Random variable", n);
      check_not_nan(function, "Rate parameter", lambda);
      check_nonnegative(function, "Rate parameter", lambda);
      check_consistent_sizes(function,
                             "Random variable", n,
                             "Rate parameter", lambda);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_rate>::value)
        return 0.0;

      // set up expression templates wrapping scalars/vecs into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_rate> lambda_vec(lambda);
      size_t size = max_size(n, lambda);

      for (size_t i = 0; i < size; i++)
        if (boost::math::isinf(lambda_vec[i]))
          return LOG_ZERO;
      for (size_t i = 0; i < size; i++)
        if (lambda_vec[i] == 0 && n_vec[i] != 0)
          return LOG_ZERO;

      // return accumulator with gradients
      OperandsAndPartials<T_rate> operands_and_partials(lambda);

      using stan::math::multiply_log;
      for (size_t i = 0; i < size; i++) {
        if (!(lambda_vec[i] == 0 && n_vec[i] == 0)) {
          if (include_summand<propto>::value)
            logp -= lgamma(n_vec[i] + 1.0);
          if (include_summand<propto, T_rate>::value)
            logp += multiply_log(n_vec[i], value_of(lambda_vec[i]))
              - value_of(lambda_vec[i]);
        }

        // gradients
        if (!is_constant_struct<T_rate>::value)
          operands_and_partials.d_x1[i]
            += n_vec[i] / value_of(lambda_vec[i]) - 1.0;
      }


      return operands_and_partials.to_var(logp, lambda);
    }

    template <typename T_n,
              typename T_rate>
    inline
    typename return_type<T_rate>::type
    poisson_log(const T_n& n, const T_rate& lambda) {
      return poisson_log<false>(n, lambda);
    }
  }
}
#endif
