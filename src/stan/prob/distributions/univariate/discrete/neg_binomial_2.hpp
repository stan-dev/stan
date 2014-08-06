#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_2_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_2_HPP

#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <boost/math/special_functions/digamma.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/internal_math.hpp>
#include <stan/prob/distributions/univariate/continuous/gamma.hpp>
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>

namespace stan {

  namespace prob {

    // NegBinomial(n|mu,phi)  [mu >= 0; phi > 0;  n >= 0]
    template <bool propto,
              typename T_n,
              typename T_location, typename T_inv_scale>
    typename return_type<T_location, T_inv_scale>::type
    neg_binomial_2_log(const T_n& n,
                     const T_location& mu,
                     const T_inv_scale& phi) {

      static const char* function = "stan::prob::neg_binomial_log(%1%)";

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

      double logp(0.0);
      check_nonnegative(function, n, "Failures variable", &logp);
      check_positive_finite(function, mu, "Location parameter", &logp);
      check_positive_finite(function, phi, "Inverse scale parameter", &logp);
      check_consistent_sizes(function,
                             n,mu,phi,
                             "Failures variable",
                             "Location parameter",
                             "Inverse scale parameter",
                             &logp);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_location,T_inv_scale>::value)
        return 0.0;

      using stan::math::multiply_log;
      using stan::math::log_sum_exp;
      using stan::math::binomial_coefficient_log;
      using boost::math::digamma;
      using boost::math::lgamma;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_location> mu_vec(mu);
      VectorView<const T_inv_scale> phi_vec(phi);
      size_t size = max_size(n, mu, phi);

      agrad::OperandsAndPartials<T_location, T_inv_scale>
        operands_and_partials(mu, phi);

      size_t len_ep = max_size(mu, phi);
      size_t len_np = max_size(n, phi);
      
      DoubleVectorView<true, is_vector<T_location>::value>
        mu__(length(mu));
      for (size_t i = 0, size = length(mu); i < size; ++i)
        mu__[i] = value_of(mu_vec[i]);
  
      DoubleVectorView<true, is_vector<T_inv_scale>::value>
        phi__(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        phi__[i] = value_of(phi_vec[i]);
      
      DoubleVectorView<true, is_vector<T_inv_scale>::value>
        log_phi(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        log_phi[i] = log(phi__[i]);

      DoubleVectorView<true, (is_vector<T_location>::value
                             || is_vector<T_inv_scale>::value)>
        log_mu_plus_phi(len_ep);
      for (size_t i = 0; i < len_ep; ++i)
        log_mu_plus_phi[i] = log(mu__[i] + phi__[i]);

      DoubleVectorView<true, (is_vector<T_n>::value
                             || is_vector<T_inv_scale>::value)>
        n_plus_phi(len_np);
      for (size_t i = 0; i < len_np; ++i)
        n_plus_phi[i] = n_vec[i] + phi__[i];

      for (size_t i = 0; i < size; i++) {
        if (include_summand<propto>::value)
          logp -= lgamma(n_vec[i] + 1.0);
        if (include_summand<propto,T_inv_scale>::value)
          logp += multiply_log(phi__[i], phi__[i]) - lgamma(phi__[i]);
        if (include_summand<propto,T_location,T_inv_scale>::value)
          logp -= (n_plus_phi[i])*log_mu_plus_phi[i];
        if (include_summand<propto,T_location>::value)
          logp += multiply_log(n_vec[i], mu__[i]);
        if (include_summand<propto,T_inv_scale>::value)
          logp += lgamma(n_plus_phi[i]);

        if (!is_constant_struct<T_location>::value)
          operands_and_partials.d_x1[i]
            += n_vec[i]/mu__[i] 
            - (n_vec[i] + phi__[i])
            / (mu__[i] + phi__[i]);
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]
            += 1.0 - n_plus_phi[i]/(mu__[i] + phi__[i])
            + log_phi[i] - log_mu_plus_phi[i] - digamma(phi__[i]) + digamma(n_plus_phi[i]);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_n,
              typename T_location, typename T_inv_scale>
    inline
    typename return_type<T_location, T_inv_scale>::type
    neg_binomial_2_log(const T_n& n,
                     const T_location& mu,
                     const T_inv_scale& phi) {
      return neg_binomial_2_log<false>(n, mu, phi);
    }


    // NegBinomial(n|eta,phi)  [phi > 0;  n >= 0]
    template <bool propto,
              typename T_n,
              typename T_log_location, typename T_inv_scale>
    typename return_type<T_log_location, T_inv_scale>::type
    neg_binomial_2_log_log(const T_n& n,
                     const T_log_location& eta,
                     const T_inv_scale& phi) {

      static const char* function = "stan::prob::neg_binomial_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive_finite;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(eta)
            && stan::length(phi)))
        return 0.0;

      double logp(0.0);
      check_nonnegative(function, n, "Failures variable", &logp);
      check_finite(function, eta, "Log location parameter", &logp);
      check_positive_finite(function, phi, "Inverse scale parameter", &logp);
      check_consistent_sizes(function,
                             n,eta,phi,
                             "Failures variable",
                             "Log location parameter",
                             "Inverse scale parameter",
                             &logp);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_log_location,T_inv_scale>::value)
        return 0.0;

      using stan::math::multiply_log;
      using stan::math::log_sum_exp;
      using stan::math::binomial_coefficient_log;
      using boost::math::digamma;
      using boost::math::lgamma;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_log_location> eta_vec(eta);
      VectorView<const T_inv_scale> phi_vec(phi);
      size_t size = max_size(n, eta, phi);

      agrad::OperandsAndPartials<T_log_location,T_inv_scale>
        operands_and_partials(eta,phi);

      size_t len_ep = max_size(eta, phi);
      size_t len_np = max_size(n, phi);

      DoubleVectorView<true, is_vector<T_log_location>::value>
        eta__(length(eta));
      for (size_t i = 0, size = length(eta); i < size; ++i)
        eta__[i] = value_of(eta_vec[i]);
  
      DoubleVectorView<true, is_vector<T_inv_scale>::value>
        phi__(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        phi__[i] = value_of(phi_vec[i]);  
        

      DoubleVectorView<true, is_vector<T_inv_scale>::value>
        log_phi(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        log_phi[i] = log(phi__[i]);

      DoubleVectorView<true, (is_vector<T_log_location>::value
                             || is_vector<T_inv_scale>::value)>
        logsumexp_eta_logphi(len_ep);
      for (size_t i = 0; i < len_ep; ++i)
        logsumexp_eta_logphi[i] = log_sum_exp(eta__[i], log_phi[i]);

      DoubleVectorView<true, (is_vector<T_n>::value
                             || is_vector<T_inv_scale>::value)>
        n_plus_phi(len_np);
      for (size_t i = 0; i < len_np; ++i)
        n_plus_phi[i] = n_vec[i] + phi__[i];

      for (size_t i = 0; i < size; i++) {
        if (include_summand<propto>::value)
          logp -= lgamma(n_vec[i] + 1.0);
        if (include_summand<propto,T_inv_scale>::value)
          logp += multiply_log(phi__[i], phi__[i]) - lgamma(phi__[i]);
        if (include_summand<propto,T_log_location,T_inv_scale>::value)
          logp -= (n_plus_phi[i])*logsumexp_eta_logphi[i];
        if (include_summand<propto,T_log_location>::value)
          logp += n_vec[i]*eta__[i];
        if (include_summand<propto,T_inv_scale>::value)
          logp += lgamma(n_plus_phi[i]);

        if (!is_constant_struct<T_log_location>::value)
          operands_and_partials.d_x1[i]
            += n_vec[i] - n_plus_phi[i]
            / (phi__[i]/exp(eta__[i]) + 1.0);
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]
            += 1.0 - n_plus_phi[i]/(exp(eta__[i]) + phi__[i])
            + log_phi[i] - logsumexp_eta_logphi[i] - digamma(phi__[i]) + digamma(n_plus_phi[i]);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_n,
              typename T_log_location, typename T_inv_scale>
    inline
    typename return_type<T_log_location, T_inv_scale>::type
    neg_binomial_2_log_log(const T_n& n,
                     const T_log_location& eta,
                     const T_inv_scale& phi) {
      return neg_binomial_2_log_log<false>(n,eta,phi);
    }

    template <class RNG>
    inline int
    neg_binomial_2_rng(const double mu,
                     const double phi,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;

      static const char* function = "stan::prob::neg_binomial_2_rng(%1%)";

      using stan::math::check_positive_finite;

      check_positive_finite(function, mu, "Location parameter", (double*)0);
      check_positive_finite(function, phi, "Inverse scale parameter", 
                            (double*)0);

      return stan::prob::poisson_rng(stan::prob::gamma_rng(phi,phi/mu,
                                                           rng),rng);
    }

    template <class RNG>
    inline int
    neg_binomial_2_log_rng(const double eta,
                     const double phi,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;

      static const char* function = "stan::prob::neg_binomial_2_log_rng(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive_finite;

      check_finite(function, eta, "Log-location parameter", (double*)0);
      check_positive_finite(function, phi, "Inverse scale parameter", 
                            (double*)0);

      return stan::prob::poisson_rng(stan::prob::gamma_rng(phi,phi/std::exp(eta),
                                                           rng),rng);
    }
  }
}
#endif
