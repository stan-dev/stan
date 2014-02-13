#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_HPP__

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

    // NegBinomial(n|alpha,beta)  [alpha > 0;  beta > 0;  n >= 0]
    template <bool propto,
              typename T_n,
              typename T_shape, typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_log(const T_n& n,
                     const T_shape& alpha,
                     const T_inv_scale& beta) {

      static const char* function = "stan::prob::neg_binomial_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(alpha)
            && stan::length(beta)))
        return 0.0;

      double logp(0.0);
      if (!check_nonnegative(function, n, "Failures variable", &logp))
        return logp;
      if (!check_finite(function, alpha, "Shape parameter", &logp))
        return logp;
      if (!check_positive(function, alpha, "Shape parameter", &logp))
        return logp;
      if (!check_finite(function, beta, "Inverse scale parameter",
                        &logp))
        return logp;
      if (!check_positive(function, beta, "Inverse scale parameter",
                          &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,alpha,beta,
                                   "Failures variable",
                                   "Shape parameter","Inverse scale parameter",
                                   &logp)))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_shape,T_inv_scale>::value)
        return 0.0;

      using stan::math::multiply_log;
      using stan::math::binomial_coefficient_log;
      using boost::math::digamma;
      using boost::math::lgamma;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);

      agrad::OperandsAndPartials<T_shape,T_inv_scale>
        operands_and_partials(alpha,beta);

      size_t len_ab = max_size(alpha,beta);
      DoubleVectorView<true,(is_vector<T_shape>::value
                             || is_vector<T_inv_scale>::value)>
        lambda(len_ab);
      for (size_t i = 0; i < len_ab; ++i)
        lambda[i] = value_of(alpha_vec[i]) / value_of(beta_vec[i]);

      DoubleVectorView<true,is_vector<T_inv_scale>::value>
        log1p_beta(length(beta));
      for (size_t i = 0; i < length(beta); ++i)
        log1p_beta[i] = log1p(value_of(beta_vec[i]));

      DoubleVectorView<true,is_vector<T_inv_scale>::value>
        log_beta_m_log1p_beta(length(beta));
      for (size_t i = 0; i < length(beta); ++i)
        log_beta_m_log1p_beta[i] = log(value_of(beta_vec[i])) - log1p_beta[i];

      DoubleVectorView<true,(is_vector<T_inv_scale>::value
                             || is_vector<T_shape>::value)>
        alpha_times_log_beta_over_1p_beta(len_ab);
      for (size_t i = 0; i < len_ab; ++i)
        alpha_times_log_beta_over_1p_beta[i]
          = value_of(alpha_vec[i])
          * log(value_of(beta_vec[i])
                / (1.0 + value_of(beta_vec[i])));

      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digamma_alpha(length(alpha));
      if (!is_constant_struct<T_shape>::value)
        for (size_t i = 0; i < length(alpha); ++i)
          digamma_alpha[i] = digamma(value_of(alpha_vec[i]));

      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_inv_scale>::value>
        log_beta(length(beta));
      if (!is_constant_struct<T_shape>::value)
        for (size_t i = 0; i < length(beta); ++i)
          log_beta[i] = log(value_of(beta_vec[i]));

      DoubleVectorView<!is_constant_struct<T_inv_scale>::value,
        (is_vector<T_shape>::value
         || is_vector<T_inv_scale>::value)>
        lambda_m_alpha_over_1p_beta(len_ab);
      if (!is_constant_struct<T_inv_scale>::value)
        for (size_t i = 0; i < len_ab; ++i)
          lambda_m_alpha_over_1p_beta[i] =
            lambda[i]
            - ( value_of(alpha_vec[i])
                / (1.0 + value_of(beta_vec[i])) );

      for (size_t i = 0; i < size; i++) {
        if (alpha_vec[i] > 1e10) { // reduces numerically to Poisson
          if (include_summand<propto>::value)
            logp -= lgamma(n_vec[i] + 1.0);
          if (include_summand<propto,T_shape,T_inv_scale>::value)
            logp += multiply_log(n_vec[i], lambda[i]) - lambda[i];

          if (!is_constant_struct<T_shape>::value)
            operands_and_partials.d_x1[i]
              += n_vec[i] / value_of(alpha_vec[i])
              - 1.0 / value_of(beta_vec[i]);
          if (!is_constant_struct<T_inv_scale>::value)
            operands_and_partials.d_x2[i]
              += (lambda[i] - n_vec[i]) / value_of(beta_vec[i]) ;
        } else { // standard density definition
          if (include_summand<propto,T_shape>::value)
            if (n_vec[i] != 0)
              logp += binomial_coefficient_log<double>(n_vec[i]
                                                       + value_of(alpha_vec[i])
                                                       - 1.0,
                                                       n_vec[i]);
          if (include_summand<propto,T_shape,T_inv_scale>::value)
            logp +=
              alpha_times_log_beta_over_1p_beta[i]
              - n_vec[i] * log1p_beta[i];

          if (!is_constant_struct<T_shape>::value)
            operands_and_partials.d_x1[i]
              += digamma(value_of(alpha_vec[i]) + n_vec[i])
              - digamma_alpha[i]
              + log_beta_m_log1p_beta[i];
          if (!is_constant_struct<T_inv_scale>::value)
            operands_and_partials.d_x2[i]
              += lambda_m_alpha_over_1p_beta[i]
              - n_vec[i]  / (value_of(beta_vec[i]) + 1.0);
        }
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_n,
              typename T_shape, typename T_inv_scale>
    inline
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_log(const T_n& n,
                     const T_shape& alpha,
                     const T_inv_scale& beta) {
      return neg_binomial_log<false>(n,alpha,beta);
    }

    // NegBinomial(n|eta,phi)  [phi > 0;  n >= 0]
    template <bool propto,
              typename T_n,
              typename T_log_location, typename T_inv_scale>
    typename return_type<T_log_location, T_inv_scale>::type
    neg_binomial_log_log(const T_n& n,
                     const T_log_location& eta,
                     const T_inv_scale& phi) {

      static const char* function = "stan::prob::neg_binomial_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(eta)
            && stan::length(phi)))
        return 0.0;

      double logp(0.0);
      if (!check_nonnegative(function, n, "Failures variable", &logp))
        return logp;
      if (!check_finite(function, eta, "Log location parameter", &logp))
        return logp;
      //if (!check_positive(function, eta, "Log location parameter", &logp))
      //  return logp;
      if (!check_finite(function, phi, "Inverse scale parameter",
                        &logp))
        return logp;
      if (!check_nonnegative(function, phi, "Inverse scale parameter",
                          &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,eta,phi,
                                   "Failures variable",
                                   "Log location parameter","Inverse scale parameter",
                                   &logp)))
        return logp;

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

      DoubleVectorView<true,is_vector<T_inv_scale>::value>
        log_phi(length(phi));
      for (size_t i = 0; i < length(phi); ++i)
        log_phi[i] = log(value_of(phi_vec[i]));

      DoubleVectorView<true,(is_vector<T_log_location>::value
                             || is_vector<T_inv_scale>::value)>
        logsumexp_eta_logphi(len_ep);
      for (size_t i = 0; i < len_ep; ++i)
        logsumexp_eta_logphi[i] = log_sum_exp(value_of(eta_vec[i]), log_phi[i]);

      DoubleVectorView<true,(is_vector<T_n>::value
                             || is_vector<T_inv_scale>::value)>
        n_plus_phi(len_np);
      for (size_t i = 0; i < len_np; ++i)
        n_plus_phi[i] = n_vec[i] + value_of(phi_vec[i]);

      for (size_t i = 0; i < size; i++) {
        if (include_summand<propto>::value)
          logp -= lgamma(n_vec[i] + 1.0);
        if (include_summand<propto,T_inv_scale>::value)
          logp += multiply_log(value_of(phi_vec[i]), value_of(phi_vec[i])) - lgamma(value_of(phi_vec[i]));
        if (include_summand<propto,T_log_location,T_inv_scale>::value)
          logp -= (n_plus_phi[i])*logsumexp_eta_logphi[i];
        if (include_summand<propto,T_log_location>::value)
          logp += n_vec[i]*value_of(eta_vec[i]);
        if (include_summand<propto,T_inv_scale>::value)
          logp += lgamma(n_plus_phi[i]);

        if (!is_constant_struct<T_log_location>::value)
          operands_and_partials.d_x1[i]
            += n_vec[i] - n_plus_phi[i]
            / (value_of(phi_vec[i])/exp(value_of(eta_vec[i])) + 1.0);
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]
            += 1.0 - n_plus_phi[i]/(exp(value_of(eta_vec[i])) + value_of(phi_vec[i]))
            + log_phi[i] - logsumexp_eta_logphi[i] - digamma(value_of(phi_vec[i])) + digamma(n_plus_phi[i]);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_n,
              typename T_log_location, typename T_inv_scale>
    inline
    typename return_type<T_log_location, T_inv_scale>::type
    neg_binomial_log_log(const T_n& n,
                     const T_log_location& eta,
                     const T_inv_scale& phi) {
      return neg_binomial_log_log<false>(n,eta,phi);
    }


    // Negative Binomial CDF
    template <typename T_n, typename T_shape,
              typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_cdf(const T_n& n, const T_shape& alpha,
                     const T_inv_scale& beta) {
      static const char* function = "stan::prob::neg_binomial_cdf(%1%)";

      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;

      // Ensure non-zero arugment lengths
      if (!(stan::length(n) && stan::length(alpha) && stan::length(beta)))
        return 1.0;

      double P(1.0);

      // Validate arguments
      if (!check_finite(function, alpha, "Shape parameter", &P))
        return P;

      if (!check_positive(function, alpha, "Shape parameter", &P))
        return P;

      if (!check_finite(function, beta, "Inverse scale parameter",
                        &P))
        return P;

      if (!check_positive(function, beta, "Inverse scale parameter",
                          &P))
        return P;

      if (!(check_consistent_sizes(function,
                                   n, alpha, beta,
                                   "Failures variable",
                                   "Shape parameter",
                                   "Inverse scale parameter",
                                   &P)))
        return P;

      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);

      // Compute vectorized CDF and gradient
      using stan::math::value_of;
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;

      using boost::math::digamma;

      agrad::OperandsAndPartials<T_shape, T_inv_scale>
        operands_and_partials(alpha, beta);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0)
          return operands_and_partials.to_var(0.0);
      }

      // Cache a few expensive function calls if alpha is a parameter
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaN_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaAlpha_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaSum_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        betaFunc_vec(stan::length(alpha));

      if (!is_constant_struct<T_shape>::value) {

        for (size_t i = 0; i < stan::length(alpha); i++) {
          const double n_dbl = value_of(n_vec[i]);
          const double alpha_dbl = value_of(alpha_vec[i]);

          digammaN_vec[i] = digamma(n_dbl + 1);
          digammaAlpha_vec[i] = digamma(alpha_dbl);
          digammaSum_vec[i] = digamma(n_dbl + alpha_dbl + 1);
          betaFunc_vec[i] = boost::math::beta(n_dbl + 1, alpha_dbl);
        }
      }

      for (size_t i = 0; i < size; i++) {

        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i])
            == std::numeric_limits<double>::infinity())
          continue;

        const double n_dbl = value_of(n_vec[i]);
        const double alpha_dbl = value_of(alpha_vec[i]);
        const double beta_dbl = value_of(beta_vec[i]);

        const double p_dbl = beta_dbl / (1.0 + beta_dbl);
        const double d_dbl = 1.0 / ( (1.0 + beta_dbl)
                                     * (1.0 + beta_dbl) );

        const double Pi = ibeta(alpha_dbl, n_dbl + 1.0, p_dbl);

        P *= Pi;

        if (!is_constant_struct<T_shape>::value) {

          double g1 = 0;
          double g2 = 0;

          stan::math::gradRegIncBeta(g1, g2, alpha_dbl,
                                     n_dbl + 1, p_dbl,
                                     digammaAlpha_vec[i],
                                     digammaN_vec[i],
                                     digammaSum_vec[i],
                                     betaFunc_vec[i]);

          operands_and_partials.d_x1[i]
            += g1 / Pi;
        }

        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]
            += d_dbl * ibeta_derivative(alpha_dbl, n_dbl + 1, p_dbl)
            / Pi;

      }

      if (!is_constant_struct<T_shape>::value)
        for(size_t i = 0; i < stan::length(alpha); ++i)
          operands_and_partials.d_x1[i] *= P;

      if (!is_constant_struct<T_inv_scale>::value)
        for(size_t i = 0; i < stan::length(beta); ++i)
          operands_and_partials.d_x2[i] *= P;

      return operands_and_partials.to_var(P);

    }

    template <typename T_n, typename T_shape,
              typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_cdf_log(const T_n& n, const T_shape& alpha,
                     const T_inv_scale& beta) {
      static const char* function = "stan::prob::neg_binomial_cdf_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;

      // Ensure non-zero arugment lengths
      if (!(stan::length(n) && stan::length(alpha) && stan::length(beta)))
        return 0.0;

      double P(0.0);

      // Validate arguments
      if (!check_finite(function, alpha, "Shape parameter", &P))
        return P;
      if (!check_positive(function, alpha, "Shape parameter", &P))
        return P;
      if (!check_finite(function, beta, "Inverse scale parameter", &P))
        return P;
      if (!check_positive(function, beta, "Inverse scale parameter", &P))
        return P;
      if (!(check_consistent_sizes(function,
                                   n, alpha, beta,
                                   "Failures variable",
                                   "Shape parameter",
                                   "Inverse scale parameter",
                                   &P)))
        return P;

      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);

      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;

      using boost::math::digamma;

      agrad::OperandsAndPartials<T_shape, T_inv_scale>
        operands_and_partials(alpha, beta);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0)
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }

      // Cache a few expensive function calls if alpha is a parameter
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaN_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaAlpha_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaSum_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        betaFunc_vec(stan::length(alpha));

      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const double n_dbl = value_of(n_vec[i]);
          const double alpha_dbl = value_of(alpha_vec[i]);

          digammaN_vec[i] = digamma(n_dbl + 1);
          digammaAlpha_vec[i] = digamma(alpha_dbl);
          digammaSum_vec[i] = digamma(n_dbl + alpha_dbl + 1);
          betaFunc_vec[i] = boost::math::beta(n_dbl + 1, alpha_dbl);
        }
      }

      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i])
            == std::numeric_limits<double>::infinity())
          continue;

        const double n_dbl = value_of(n_vec[i]);
        const double alpha_dbl = value_of(alpha_vec[i]);
        const double beta_dbl = value_of(beta_vec[i]);
        const double p_dbl = beta_dbl / (1.0 + beta_dbl);
        const double d_dbl = 1.0 / ( (1.0 + beta_dbl)
                                     * (1.0 + beta_dbl) );
        const double Pi = ibeta(alpha_dbl, n_dbl + 1.0, p_dbl);

        P += log(Pi);

        if (!is_constant_struct<T_shape>::value) {
          double g1 = 0;
          double g2 = 0;

          stan::math::gradRegIncBeta(g1, g2, alpha_dbl,
                                     n_dbl + 1, p_dbl,
                                     digammaAlpha_vec[i],
                                     digammaN_vec[i],
                                     digammaSum_vec[i],
                                     betaFunc_vec[i]);
          operands_and_partials.d_x1[i] += g1 / Pi;
        }
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]
            += d_dbl * ibeta_derivative(alpha_dbl, n_dbl + 1, p_dbl) / Pi;
      }

      return operands_and_partials.to_var(P);
    }

    template <typename T_n, typename T_shape,
              typename T_inv_scale>
    typename return_type<T_shape, T_inv_scale>::type
    neg_binomial_ccdf_log(const T_n& n, const T_shape& alpha,
                     const T_inv_scale& beta) {
      static const char* function = "stan::prob::neg_binomial_ccdf_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;

      // Ensure non-zero arugment lengths
      if (!(stan::length(n) && stan::length(alpha) && stan::length(beta)))
        return 0.0;

      double P(0.0);

      // Validate arguments
      if (!check_finite(function, alpha, "Shape parameter", &P))
        return P;
      if (!check_positive(function, alpha, "Shape parameter", &P))
        return P;
      if (!check_finite(function, beta, "Inverse scale parameter", &P))
        return P;
      if (!check_positive(function, beta, "Inverse scale parameter", &P))
        return P;
      if (!(check_consistent_sizes(function,
                                   n, alpha, beta,
                                   "Failures variable",
                                   "Shape parameter",
                                   "Inverse scale parameter",
                                   &P)))
        return P;

      // Wrap arguments in vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t size = max_size(n, alpha, beta);

      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;

      using boost::math::digamma;

      agrad::OperandsAndPartials<T_shape, T_inv_scale>
        operands_and_partials(alpha, beta);

      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) <= 0)
          return operands_and_partials.to_var(0.0);
      }

      // Cache a few expensive function calls if alpha is a parameter
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaN_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaAlpha_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        digammaSum_vec(stan::length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,
        is_vector<T_shape>::value>
        betaFunc_vec(stan::length(alpha));

      if (!is_constant_struct<T_shape>::value) {
        for (size_t i = 0; i < stan::length(alpha); i++) {
          const double n_dbl = value_of(n_vec[i]);
          const double alpha_dbl = value_of(alpha_vec[i]);

          digammaN_vec[i] = digamma(n_dbl + 1);
          digammaAlpha_vec[i] = digamma(alpha_dbl);
          digammaSum_vec[i] = digamma(n_dbl + alpha_dbl + 1);
          betaFunc_vec[i] = boost::math::beta(n_dbl + 1, alpha_dbl);
        }
      }

      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i])
            == std::numeric_limits<double>::infinity())
          return operands_and_partials.to_var(stan::math::negative_infinity());

        const double n_dbl = value_of(n_vec[i]);
        const double alpha_dbl = value_of(alpha_vec[i]);
        const double beta_dbl = value_of(beta_vec[i]);
        const double p_dbl = beta_dbl / (1.0 + beta_dbl);
        const double d_dbl = 1.0 / ( (1.0 + beta_dbl)
                                     * (1.0 + beta_dbl) );
        const double Pi = 1.0 - ibeta(alpha_dbl, n_dbl + 1.0, p_dbl);

        P += log(Pi);

        if (!is_constant_struct<T_shape>::value) {
          double g1 = 0;
          double g2 = 0;

          stan::math::gradRegIncBeta(g1, g2, alpha_dbl,
                                     n_dbl + 1, p_dbl,
                                     digammaAlpha_vec[i],
                                     digammaN_vec[i],
                                     digammaSum_vec[i],
                                     betaFunc_vec[i]);
          operands_and_partials.d_x1[i] -= g1 / Pi;
        }
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[i]
            -= d_dbl * ibeta_derivative(alpha_dbl, n_dbl + 1, p_dbl) / Pi;
      }

      return operands_and_partials.to_var(P);
    }

    template <class RNG>
    inline int
    neg_binomial_rng(const double alpha,
                     const double beta,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;

      static const char* function = "stan::prob::neg_binomial_rng(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;

      if (!check_finite(function, alpha, "Shape parameter"))
        return 0;
      if (!check_positive(function, alpha, "Shape parameter"))
        return 0;
      if (!check_finite(function, beta, "Inverse scale parameter"))
        return 0;
      if (!check_positive(function, beta, "Inverse scale parameter"))
        return 0;

      return stan::prob::poisson_rng(stan::prob::gamma_rng(alpha,1.0 / beta,
                                                           rng),rng);
    }

    template <class RNG>
    inline int
    neg_binomial_log_rng(const double eta,
                     const double phi,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;

      static const char* function = "stan::prob::neg_binomial_rng(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;

      if (!check_finite(function, eta, "Log-location parameter"))
        return 0;
      if (!check_finite(function, phi, "Inverse scale parameter"))
        return 0;
      if (!check_positive(function, phi, "Inverse scale parameter"))
        return 0;

        return stan::prob::poisson_rng(stan::prob::gamma_rng(phi,phi/std::exp(eta),
                                                           rng),rng);

      //variate_generator<RNG&, negative_binomial_distribution<> >
      //  negative_binomial_rng(rng, negative_binomial_distribution<>(phi, phi/(phi+std::exp(eta))));
      //return negative_binomial_rng();
    }
  }
}
#endif
