#ifndef STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_2_LOG_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_2_LOG_LOG_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <cmath>

namespace stan {

  namespace math {

    // NegBinomial(n|eta, phi)  [phi > 0;  n >= 0]
    template <bool propto,
              typename T_n,
              typename T_log_location, typename T_precision>
    typename return_type<T_log_location, T_precision>::type
    neg_binomial_2_log_log(const T_n& n,
                           const T_log_location& eta,
                           const T_precision& phi) {
      typedef typename stan::partials_return_type<T_n, T_log_location,
                                                  T_precision>::type
        T_partials_return;

      static const char* function("stan::prob::neg_binomial_2_log_log");

      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive_finite;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::math::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(eta)
            && stan::length(phi)))
        return 0.0;

      T_partials_return logp(0.0);
      check_nonnegative(function, "Failures variable", n);
      check_finite(function, "Log location parameter", eta);
      check_positive_finite(function, "Precision parameter", phi);
      check_consistent_sizes(function,
                             "Failures variable", n,
                             "Log location parameter", eta,
                             "Precision parameter", phi);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_log_location, T_precision>::value)
        return 0.0;

      using stan::math::multiply_log;
      using stan::math::log_sum_exp;
      using stan::math::digamma;
      using stan::math::lgamma;
      using std::exp;
      using std::log;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_log_location> eta_vec(eta);
      VectorView<const T_precision> phi_vec(phi);
      size_t size = max_size(n, eta, phi);

      OperandsAndPartials<T_log_location, T_precision>
        operands_and_partials(eta, phi);

      size_t len_ep = max_size(eta, phi);
      size_t len_np = max_size(n, phi);

      VectorBuilder<true, T_partials_return, T_log_location> eta__(length(eta));
      for (size_t i = 0, size = length(eta); i < size; ++i)
        eta__[i] = value_of(eta_vec[i]);

      VectorBuilder<true, T_partials_return, T_precision> phi__(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        phi__[i] = value_of(phi_vec[i]);


      VectorBuilder<true, T_partials_return, T_precision>
        log_phi(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        log_phi[i] = log(phi__[i]);

      VectorBuilder<true, T_partials_return, T_log_location, T_precision>
        logsumexp_eta_logphi(len_ep);
      for (size_t i = 0; i < len_ep; ++i)
        logsumexp_eta_logphi[i] = log_sum_exp(eta__[i], log_phi[i]);

      VectorBuilder<true, T_partials_return, T_n, T_precision>
        n_plus_phi(len_np);
      for (size_t i = 0; i < len_np; ++i)
        n_plus_phi[i] = n_vec[i] + phi__[i];

      for (size_t i = 0; i < size; i++) {
        if (include_summand<propto>::value)
          logp -= lgamma(n_vec[i] + 1.0);
        if (include_summand<propto, T_precision>::value)
          logp += multiply_log(phi__[i], phi__[i]) - lgamma(phi__[i]);
        if (include_summand<propto, T_log_location, T_precision>::value)
          logp -= (n_plus_phi[i])*logsumexp_eta_logphi[i];
        if (include_summand<propto, T_log_location>::value)
          logp += n_vec[i]*eta__[i];
        if (include_summand<propto, T_precision>::value)
          logp += lgamma(n_plus_phi[i]);

        if (!is_constant_struct<T_log_location>::value)
          operands_and_partials.d_x1[i]
            += n_vec[i] - n_plus_phi[i]
            / (phi__[i]/exp(eta__[i]) + 1.0);
        if (!is_constant_struct<T_precision>::value)
          operands_and_partials.d_x2[i]
            += 1.0 - n_plus_phi[i]/(exp(eta__[i]) + phi__[i])
            + log_phi[i] - logsumexp_eta_logphi[i] - digamma(phi__[i])
            + digamma(n_plus_phi[i]);
      }
      return operands_and_partials.to_var(logp, eta, phi);
    }

    template <typename T_n,
              typename T_log_location, typename T_precision>
    inline
    typename return_type<T_log_location, T_precision>::type
    neg_binomial_2_log_log(const T_n& n,
                           const T_log_location& eta,
                           const T_precision& phi) {
      return neg_binomial_2_log_log<false>(n, eta, phi);
    }
  }
}
#endif
