#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BINOMIAL_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BINOMIAL_HPP

#include <boost/random/binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_greater_or_equal.hpp>
#include <stan/error_handling/scalar/check_less_or_equal.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/inv_logit.hpp>
#include <stan/math/functions/log1m.hpp>
#include <stan/math/functions/log_inv_logit.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

#include <stan/math/functions/binomial_coefficient_log.hpp>

namespace stan {

  namespace prob {

    // Binomial(n|N,theta)  [N >= 0;  0 <= n <= N;  0 <= theta <= 1]
    template <bool propto,
              typename T_n,
              typename T_N,
              typename T_prob>
    typename return_type<T_prob>::type
    binomial_log(const T_n& n, 
                 const T_N& N, 
                 const T_prob& theta) {

      static const std::string function("stan::prob::binomial_log");
      
      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(N)
            && stan::length(theta)))
        return 0.0;

      double logp = 0;
      check_bounded(function, "Successes variable", n, 0, N);
      check_nonnegative(function, "Population size parameter", N);
      check_finite(function, "Probability parameter", theta);
      check_bounded(function, "Probability parameter", theta, 0.0, 1.0);
      check_consistent_sizes(function,
                             "Successes variable", n,
                             "Population size parameter", N,
                             "Probability parameter", theta);


      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_prob>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_prob> theta_vec(theta);
      size_t size = max_size(n, N, theta);

      agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);
      
      using stan::math::multiply_log;
      using stan::math::binomial_coefficient_log;
      using stan::math::log1m;
        
      if (include_summand<propto>::value)
        for (size_t i = 0; i < size; ++i)
          logp += binomial_coefficient_log(N_vec[i],n_vec[i]);

      DoubleVectorView<true,is_vector<T_prob>::value> log1m_theta(length(theta));
      for (size_t i = 0; i < length(theta); ++i)
        log1m_theta[i] = log1m(value_of(theta_vec[i]));

      // no test for include_summand because return if not live
      for (size_t i = 0; i < size; ++i)
        logp += multiply_log(n_vec[i],value_of(theta_vec[i]))
          + (N_vec[i] - n_vec[i]) * log1m_theta[i];

      if (length(theta) == 1) {
        double temp1 = 0;
        double temp2 = 0;
        for (size_t i = 0; i < size; ++i) {
          temp1 += n_vec[i];
          temp2 += N_vec[i] - n_vec[i];
        }
        if (!is_constant_struct<T_prob>::value) {
          operands_and_partials.d_x1[0] 
            += temp1 / value_of(theta_vec[0])
            - temp2 / (1.0 - value_of(theta_vec[0]));
        }
      } else {
        if (!is_constant_struct<T_prob>::value) {
          for (size_t i = 0; i < size; ++i)
            operands_and_partials.d_x1[i] 
              += n_vec[i] / value_of(theta_vec[i])
              - (N_vec[i] - n_vec[i]) / (1.0 - value_of(theta_vec[i]));
        }
      }

      return operands_and_partials.to_var(logp);
    }

    template <typename T_n, 
              typename T_N,
              typename T_prob>
    inline
    typename return_type<T_prob>::type
    binomial_log(const T_n& n, 
                 const T_N& N, 
                 const T_prob& theta) {
      return binomial_log<false>(n,N,theta);
    }

    // BinomialLogit(n|N,alpha)  [N >= 0;  0 <= n <= N]
    // BinomialLogit(n|N,alpha) = Binomial(n|N,inv_logit(alpha))
    template <bool propto,
              typename T_n,
              typename T_N,
              typename T_prob>
    typename return_type<T_prob>::type
    binomial_logit_log(const T_n& n, 
                       const T_N& N, 
                       const T_prob& alpha) {

      static const std::string function("stan::prob::binomial_logit_log");
      
      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(N)
            && stan::length(alpha)))
        return 0.0;

      double logp = 0;
      check_bounded(function, "Successes variable", n, 0, N);
      check_nonnegative(function, "Population size parameter", N);
      check_finite(function, "Probability parameter", alpha);
      check_consistent_sizes(function,
                             "Successes variable", n,
                             "Population size parameter", N,
                             "Probability parameter", alpha);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_prob>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_prob> alpha_vec(alpha);
      size_t size = max_size(n, N, alpha);

      agrad::OperandsAndPartials<T_prob> operands_and_partials(alpha);
      
      using stan::math::binomial_coefficient_log;
      using stan::math::log_inv_logit;
      using stan::math::inv_logit;
        
      if (include_summand<propto>::value)
        for (size_t i = 0; i < size; ++i)
          logp += binomial_coefficient_log(N_vec[i],n_vec[i]);

      DoubleVectorView<true,is_vector<T_prob>::value> log_inv_logit_alpha(length(alpha));
      for (size_t i = 0; i < length(alpha); ++i)
        log_inv_logit_alpha[i] = log_inv_logit(value_of(alpha_vec[i]));

      DoubleVectorView<true,is_vector<T_prob>::value> log_inv_logit_neg_alpha(length(alpha));
      for (size_t i = 0; i < length(alpha); ++i)
        log_inv_logit_neg_alpha[i] = log_inv_logit(-value_of(alpha_vec[i]));

      for (size_t i = 0; i < size; ++i)
        logp += n_vec[i] * log_inv_logit_alpha[i]
          + (N_vec[i] - n_vec[i]) * log_inv_logit_neg_alpha[i];

      if (length(alpha) == 1) {
        double temp1 = 0;
        double temp2 = 0;
        for (size_t i = 0; i < size; ++i) {
          temp1 += n_vec[i];
          temp2 += N_vec[i] - n_vec[i];
        }
        if (!is_constant_struct<T_prob>::value) {
          operands_and_partials.d_x1[0] 
            += temp1 * inv_logit(-value_of(alpha_vec[0]))
            - temp2 * inv_logit(value_of(alpha_vec[0]));
        }
      } else {
        if (!is_constant_struct<T_prob>::value) {
          for (size_t i = 0; i < size; ++i)
            operands_and_partials.d_x1[i] 
              += n_vec[i] * inv_logit(-value_of(alpha_vec[i]))
              - (N_vec[i] - n_vec[i]) * inv_logit(value_of(alpha_vec[i]));
        }
      }

      return operands_and_partials.to_var(logp);
    }

    template <typename T_n, 
              typename T_N,
              typename T_prob>
    inline
    typename return_type<T_prob>::type
    binomial_logit_log(const T_n& n, 
                       const T_N& N, 
                       const T_prob& alpha) {
      return binomial_logit_log<false>(n,N,alpha);
    }


    // Binomial CDF
    template <typename T_n, typename T_N, typename T_prob>
    typename return_type<T_prob>::type
    binomial_cdf(const T_n& n, const T_N& N, const T_prob& theta) {
          
      static const std::string function("stan::prob::binomial_cdf");
          
      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero arguments lenghts
      if (!(stan::length(n) && stan::length(N) && stan::length(theta)))
        return 1.0;
          
      double P(1.0);
          
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
          
      // Compute vectorized CDF and gradient
      using stan::math::value_of;
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;
          
      agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0) 
          return operands_and_partials.to_var(0.0);
      }
        
      for (size_t i = 0; i < size; i++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= value_of(N_vec[i])) {
          continue;
        }
          
        const double n_dbl = value_of(n_vec[i]);
        const double N_dbl = value_of(N_vec[i]);
        const double theta_dbl = value_of(theta_vec[i]);

        const double Pi = ibeta(N_dbl - n_dbl, n_dbl + 1, 1 - theta_dbl);
          
        P *= Pi;

        if (!is_constant_struct<T_prob>::value)
          operands_and_partials.d_x1[i] 
            += - ibeta_derivative(N_dbl - n_dbl, n_dbl + 1, 1 - theta_dbl) / Pi;
          
              
      }
          
      if (!is_constant_struct<T_prob>::value) {
        for(size_t i = 0; i < stan::length(theta); ++i) operands_and_partials.d_x1[i] *= P;
      }
          
      return operands_and_partials.to_var(P);
        
    }

    template <typename T_n, typename T_N, typename T_prob>
    typename return_type<T_prob>::type
    binomial_cdf_log(const T_n& n, const T_N& N, const T_prob& theta) {
          
      static const std::string function("stan::prob::binomial_cdf_log");
          
      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero arguments lenghts
      if (!(stan::length(n) && stan::length(N) && stan::length(theta)))
        return 0.0;
          
      double P(0.0);
          
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
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;
          
      agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as negative infinity
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
        
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= value_of(N_vec[i])) {
          continue;
        }
        const double n_dbl = value_of(n_vec[i]);
        const double N_dbl = value_of(N_vec[i]);
        const double theta_dbl = value_of(theta_vec[i]);
        const double Pi = ibeta(N_dbl - n_dbl, n_dbl + 1, 1 - theta_dbl);
          
        P += log(Pi);

        if (!is_constant_struct<T_prob>::value)
          operands_and_partials.d_x1[i] 
            += - ibeta_derivative(N_dbl - n_dbl, n_dbl + 1, 1 - theta_dbl) / Pi;
      }
          
      return operands_and_partials.to_var(P);
    }

    template <typename T_n, typename T_N, typename T_prob>
    typename return_type<T_prob>::type
    binomial_ccdf_log(const T_n& n, const T_N& N, const T_prob& theta) {
          
      static const std::string function("stan::prob::binomial_ccdf_log");
          
      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero arguments lenghts
      if (!(stan::length(n) && stan::length(N) && stan::length(theta)))
        return 0.0;
          
      double P(0.0);
          
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
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;
          
      agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as negative infinity
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0) 
          return operands_and_partials.to_var(0.0);
      }
        
      for (size_t i = 0; i < size; i++) {
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= value_of(N_vec[i])) {
          return operands_and_partials.to_var(stan::math::negative_infinity());
        }
        const double n_dbl = value_of(n_vec[i]);
        const double N_dbl = value_of(N_vec[i]);
        const double theta_dbl = value_of(theta_vec[i]);
        const double Pi = 1.0 - ibeta(N_dbl - n_dbl, n_dbl + 1, 1 - theta_dbl);
          
        P += log(Pi);

        if (!is_constant_struct<T_prob>::value)
          operands_and_partials.d_x1[i] 
            += ibeta_derivative(N_dbl - n_dbl, n_dbl + 1, 1 - theta_dbl) / Pi;
      }
          
      return operands_and_partials.to_var(P);
    }


    template <class RNG>
    inline int
    binomial_rng(const int N,
                 const double theta,
                 RNG& rng) {
      using boost::variate_generator;
      using boost::binomial_distribution;

      static const std::string function("stan::prob::binomial_rng");
      
      using stan::error_handling::check_finite;
      using stan::error_handling::check_less_or_equal;
      using stan::error_handling::check_greater_or_equal;
      using stan::error_handling::check_nonnegative;

      check_nonnegative(function, "Population size parameter", N);
      check_finite(function, "Probability parameter", theta);
      check_less_or_equal(function, "Probability parameter", theta, 1.0);
      check_greater_or_equal(function, "Probability parameter", theta, 0.0);

      variate_generator<RNG&, binomial_distribution<> >
        binomial_rng(rng, binomial_distribution<>(N, theta));
      return binomial_rng();
    }
    
  }
}
#endif
