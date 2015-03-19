#ifndef STAN__MATH__PRIM__SCAL__PROB__NEG_BINOMIAL_2_LOG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__NEG_BINOMIAL_2_LOG_HPP

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
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>

namespace stan {

  namespace prob {

    // NegBinomial(n|mu,phi)  [mu >= 0; phi > 0;  n >= 0]
    template <bool propto,
              typename T_n,
              typename T_location, typename T_precision>
    typename return_type<T_location, T_precision>::type
    neg_binomial_2_log(const T_n& n,
                       const T_location& mu,
                       const T_precision& phi) {
      typedef typename stan::partials_return_type<T_n,T_location,
                                                  T_precision>::type
        T_partials_return;

      static const char* function("stan::prob::neg_binomial_2_log");

      using stan::math::check_positive_finite;
      using stan::math::check_nonnegative;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(mu)
            && stan::length(phi)))
        return 0.0;

      T_partials_return logp(0.0);
      check_nonnegative(function, "Failures variable", n);
      check_positive_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Precision parameter", phi);
      check_consistent_sizes(function,
                             "Failures variable", n,
                             "Location parameter", mu,
                             "Precision parameter", phi);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_location,T_precision>::value)
        return 0.0;

      using stan::math::multiply_log;
      using stan::math::digamma;
      using stan::math::lgamma;
      using std::log;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_location> mu_vec(mu);
      VectorView<const T_precision> phi_vec(phi);
      size_t size = max_size(n, mu, phi);

      agrad::OperandsAndPartials<T_location, T_precision>
        operands_and_partials(mu, phi);

      size_t len_ep = max_size(mu, phi);
      size_t len_np = max_size(n, phi);

      VectorBuilder<true, T_partials_return, T_location> mu__(length(mu));
      for (size_t i = 0, size = length(mu); i < size; ++i)
        mu__[i] = value_of(mu_vec[i]);

      VectorBuilder<true, T_partials_return, T_precision> phi__(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        phi__[i] = value_of(phi_vec[i]);

      VectorBuilder<true, T_partials_return, T_precision> log_phi(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        log_phi[i] = log(phi__[i]);

      VectorBuilder<true, T_partials_return, T_location, T_precision>
        log_mu_plus_phi(len_ep);
      for (size_t i = 0; i < len_ep; ++i)
        log_mu_plus_phi[i] = log(mu__[i] + phi__[i]);

      VectorBuilder<true, T_partials_return, T_n, T_precision>
        n_plus_phi(len_np);
      for (size_t i = 0; i < len_np; ++i)
        n_plus_phi[i] = n_vec[i] + phi__[i];

      for (size_t i = 0; i < size; i++) {
        if (include_summand<propto>::value)
          logp -= lgamma(n_vec[i] + 1.0);
        if (include_summand<propto,T_precision>::value)
          logp += multiply_log(phi__[i], phi__[i]) - lgamma(phi__[i]);
        if (include_summand<propto,T_location,T_precision>::value)
          logp -= (n_plus_phi[i])*log_mu_plus_phi[i];
        if (include_summand<propto,T_location>::value)
          logp += multiply_log(n_vec[i], mu__[i]);
        if (include_summand<propto,T_precision>::value)
          logp += lgamma(n_plus_phi[i]);

        if (!is_constant_struct<T_location>::value)
          operands_and_partials.d_x1[i]
            += n_vec[i]/mu__[i]
            - (n_vec[i] + phi__[i])
            / (mu__[i] + phi__[i]);
        if (!is_constant_struct<T_precision>::value)
          operands_and_partials.d_x2[i]
            += 1.0 - n_plus_phi[i]/(mu__[i] + phi__[i])
            + log_phi[i] - log_mu_plus_phi[i] - digamma(phi__[i])
            + digamma(n_plus_phi[i]);
      }
      return operands_and_partials.to_var(logp,mu,phi);
    }

    template <typename T_n,
              typename T_location, typename T_precision>
    inline
    typename return_type<T_location, T_precision>::type
    neg_binomial_2_log(const T_n& n,
                       const T_location& mu,
                       const T_precision& phi) {
      return neg_binomial_2_log<false>(n, mu, phi);
    }
  }
}
#endif
