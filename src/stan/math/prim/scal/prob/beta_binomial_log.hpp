#ifndef STAN_MATH_PRIM_SCAL_PROB_BETA_BINOMIAL_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_BETA_BINOMIAL_LOG_HPP

#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/contains_nonconstant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/prob/beta_rng.hpp>
#include <stan/math/prim/scal/fun/F32.hpp>
#include <stan/math/prim/scal/fun/grad_F32.hpp>

namespace stan {

  namespace math {

    // BetaBinomial(n|alpha, beta) [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto,
              typename T_n, typename T_N,
              typename T_size1, typename T_size2>
    typename return_type<T_size1, T_size2>::type
    beta_binomial_log(const T_n& n,
                      const T_N& N,
                      const T_size1& alpha,
                      const T_size2& beta) {
      static const char* function("stan::math::beta_binomial_log");
      typedef typename stan::partials_return_type<T_size1, T_size2>::type
        T_partials_return;

      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::math::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(N)
            && stan::length(alpha)
            && stan::length(beta)))
        return 0.0;

      T_partials_return logp(0.0);
      check_nonnegative(function, "Population size parameter", N);
      check_positive_finite(function,
                            "First prior sample size parameter", alpha);
      check_positive_finite(function,
                            "Second prior sample size parameter", beta);
      check_consistent_sizes(function,
                             "Successes variable", n,
                             "Population size parameter", N,
                             "First prior sample size parameter", alpha,
                             "Second prior sample size parameter", beta);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_size1, T_size2>::value)
        return 0.0;

      OperandsAndPartials<T_size1, T_size2>
        operands_and_partials(alpha, beta);

      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_size1> alpha_vec(alpha);
      VectorView<const T_size2> beta_vec(beta);
      size_t size = max_size(n, N, alpha, beta);

      for (size_t i = 0; i < size; i++) {
        if (n_vec[i] < 0 || n_vec[i] > N_vec[i])
          return operands_and_partials.to_var(LOG_ZERO, alpha, beta);
      }

      using stan::math::lbeta;
      using stan::math::binomial_coefficient_log;
      using stan::math::digamma;

      VectorBuilder<include_summand<propto>::value,
                    T_partials_return, T_n, T_N>
        normalizing_constant(max_size(N, n));
      for (size_t i = 0; i < max_size(N, n); i++)
        if (include_summand<propto>::value)
          normalizing_constant[i]
            = binomial_coefficient_log(N_vec[i], n_vec[i]);

      VectorBuilder<include_summand<propto, T_size1, T_size2>::value,
                    T_partials_return, T_n, T_N, T_size1, T_size2>
        lbeta_numerator(size);
      for (size_t i = 0; i < size; i++)
        if (include_summand<propto, T_size1, T_size2>::value)
          lbeta_numerator[i] = lbeta(n_vec[i] + value_of(alpha_vec[i]),
                                     N_vec[i] - n_vec[i]
                                     + value_of(beta_vec[i]));

      VectorBuilder<include_summand<propto, T_size1, T_size2>::value,
                    T_partials_return, T_size1, T_size2>
        lbeta_denominator(max_size(alpha, beta));
      for (size_t i = 0; i < max_size(alpha, beta); i++)
        if (include_summand<propto, T_size1, T_size2>::value)
          lbeta_denominator[i] = lbeta(value_of(alpha_vec[i]),
                                       value_of(beta_vec[i]));

      VectorBuilder<!is_constant_struct<T_size1>::value,
                    T_partials_return, T_n, T_size1>
        digamma_n_plus_alpha(max_size(n, alpha));
      for (size_t i = 0; i < max_size(n, alpha); i++)
        if (!is_constant_struct<T_size1>::value)
          digamma_n_plus_alpha[i]
            = digamma(n_vec[i] + value_of(alpha_vec[i]));

      VectorBuilder<contains_nonconstant_struct<T_size1, T_size2>::value,
                    T_partials_return, T_N, T_size1, T_size2>
        digamma_N_plus_alpha_plus_beta(max_size(N, alpha, beta));
      for (size_t i = 0; i < max_size(N, alpha, beta); i++)
        if (contains_nonconstant_struct<T_size1, T_size2>::value)
          digamma_N_plus_alpha_plus_beta[i]
            = digamma(N_vec[i] + value_of(alpha_vec[i])
                      + value_of(beta_vec[i]));

      VectorBuilder<contains_nonconstant_struct<T_size1, T_size2>::value,
                    T_partials_return, T_size1, T_size2>
        digamma_alpha_plus_beta(max_size(alpha, beta));
      for (size_t i = 0; i < max_size(alpha, beta); i++)
        if (contains_nonconstant_struct<T_size1, T_size2>::value)
          digamma_alpha_plus_beta[i]
            = digamma(value_of(alpha_vec[i]) + value_of(beta_vec[i]));

      VectorBuilder<!is_constant_struct<T_size1>::value,
                    T_partials_return, T_size1> digamma_alpha(length(alpha));
      for (size_t i = 0; i < length(alpha); i++)
        if (!is_constant_struct<T_size1>::value)
          digamma_alpha[i] = digamma(value_of(alpha_vec[i]));

      VectorBuilder<!is_constant_struct<T_size2>::value,
                    T_partials_return, T_size2>
        digamma_beta(length(beta));
      for (size_t i = 0; i < length(beta); i++)
        if (!is_constant_struct<T_size2>::value)
          digamma_beta[i] = digamma(value_of(beta_vec[i]));

      for (size_t i = 0; i < size; i++) {
        if (include_summand<propto>::value)
          logp += normalizing_constant[i];
        if (include_summand<propto, T_size1, T_size2>::value)
          logp += lbeta_numerator[i] - lbeta_denominator[i];

        if (!is_constant_struct<T_size1>::value)
          operands_and_partials.d_x1[i]
            += digamma_n_plus_alpha[i]
            - digamma_N_plus_alpha_plus_beta[i]
            + digamma_alpha_plus_beta[i]
            - digamma_alpha[i];
        if (!is_constant_struct<T_size2>::value)
          operands_and_partials.d_x2[i]
            += digamma(value_of(N_vec[i]-n_vec[i]+beta_vec[i]))
            - digamma_N_plus_alpha_plus_beta[i]
            + digamma_alpha_plus_beta[i]
            - digamma_beta[i];
      }
      return operands_and_partials.to_var(logp, alpha, beta);
    }

    template <typename T_n,
              typename T_N,
              typename T_size1,
              typename T_size2>
    typename return_type<T_size1, T_size2>::type
    beta_binomial_log(const T_n& n, const T_N& N,
                      const T_size1& alpha, const T_size2& beta) {
      return beta_binomial_log<false>(n, N, alpha, beta);
    }

  }
}
#endif
