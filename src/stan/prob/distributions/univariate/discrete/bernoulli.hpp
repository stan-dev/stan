#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BERNOULLI_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BERNOULLI_HPP

#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/inv_logit.hpp>
#include <stan/math/functions/log1m.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // Bernoulli(n|theta)   [0 <= n <= 1;   0 <= theta <= 1]
    // FIXME: documentation
    template <bool propto, typename T_n, typename T_prob>
    typename return_type<T_prob>::type
    bernoulli_log(const T_n& n,
                  const T_prob& theta) {
      static const std::string function("stan::prob::bernoulli_log");

      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::math::log1m;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(theta)))
        return 0.0;
      
      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_bounded(function, "n", n, 0, 1);
      check_finite(function, "Probability parameter", theta);
      check_bounded(function, "Probability parameter", theta, 0.0, 1.0);
      check_consistent_sizes(function,
                             "Random variable", n,
                             "Probability parameter", theta);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_prob>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_prob> theta_vec(theta);
      size_t N = max_size(n, theta);
      agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);
      
      if (length(theta) == 1) {  
        size_t sum = 0;
        for (size_t n = 0; n < N; n++) {
          sum += value_of(n_vec[n]);
        }
        const double theta_dbl = value_of(theta_vec[0]);
        // avoid nans when sum == N or sum == 0
        if (sum == N) {
          logp += N * log(theta_dbl);
          if (!is_constant_struct<T_prob>::value)
            operands_and_partials.d_x1[0] += N / theta_dbl;
        } else if (sum == 0) {
          logp += N * log1m(theta_dbl);
          if (!is_constant_struct<T_prob>::value)
            operands_and_partials.d_x1[0] += N / (theta_dbl - 1);
        } else {
          const double log_theta = log(theta_dbl);
          const double log1m_theta = log1m(theta_dbl);

          logp += sum * log_theta;
          logp += (N - sum) * log1m_theta;

          // gradient
          if (!is_constant_struct<T_prob>::value) {
            operands_and_partials.d_x1[0] += sum / theta_dbl;
            operands_and_partials.d_x1[0] += (N - sum) / (theta_dbl - 1);
          }
        }
      } else {
        for (size_t n = 0; n < N; n++) {
          // pull out values of arguments
          const int n_int = value_of(n_vec[n]);
          const double theta_dbl = value_of(theta_vec[n]);
    
          if (n_int == 1)
            logp += log(theta_dbl);
          else
            logp += log1m(theta_dbl);
    
          // gradient
          if (!is_constant_struct<T_prob>::value) {
            if (n_int == 1)
              operands_and_partials.d_x1[n] += 1.0 / theta_dbl;
            else
              operands_and_partials.d_x1[n] += 1.0 / (theta_dbl - 1);
          }
        }
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_prob>
    inline
    typename return_type<T_prob>::type
    bernoulli_log(const T_y& n, 
                  const T_prob& theta) {
      return bernoulli_log<false>(n,theta);
    }


    // Bernoulli(n|inv_logit(theta))   [0 <= n <= 1;   -inf <= theta <= inf]
    // FIXME: documentation
    template <bool propto, typename T_n, typename T_prob>
    typename return_type<T_prob>::type
    bernoulli_logit_log(const T_n& n, const T_prob& theta) {
      static const std::string function("stan::prob::bernoulli_logit_log");

      using stan::is_constant_struct;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_bounded;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
      using stan::math::log1p;
      using stan::math::inv_logit;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(theta)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);
      
      // validate args (here done over var, which should be OK)
      check_bounded(function, "n", n, 0, 1);
      check_not_nan(function, "Logit transformed probability parameter", theta);
      check_consistent_sizes(function,
                             "Random variable", n,
                             "Probability parameter", theta);
      
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_prob>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_prob> theta_vec(theta);
      size_t N = max_size(n, theta);
      agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const int n_int = value_of(n_vec[n]);
        const double theta_dbl = value_of(theta_vec[n]);

        // reusable subexpression values
        const int sign = 2*n_int-1;
        const double ntheta = sign * theta_dbl;
        const double exp_m_ntheta = exp(-ntheta);
  
        // Handle extreme values gracefully using Taylor approximations.
        const static double cutoff = 20.0;
        if (ntheta > cutoff)
          logp -= exp_m_ntheta;
        else if (ntheta < -cutoff)
          logp += ntheta;
        else
          logp -= log1p(exp_m_ntheta);

        // gradients
        if (!is_constant_struct<T_prob>::value) {
          const static double cutoff = 20.0;
          if (ntheta > cutoff)
            operands_and_partials.d_x1[n] -= exp_m_ntheta;
          else if (ntheta < -cutoff)
            operands_and_partials.d_x1[n] += sign;
          else
            operands_and_partials.d_x1[n] += sign * exp_m_ntheta / (exp_m_ntheta + 1);
        }
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_n,
              typename T_prob>
    inline
    typename return_type<T_prob>::type
    bernoulli_logit_log(const T_n& n, 
                        const T_prob& theta) {
      return bernoulli_logit_log<false>(n,theta);
    }
      
    // Bernoulli CDF
    template <typename T_n, typename T_prob>
    typename return_type<T_prob>::type
    bernoulli_cdf(const T_n& n, const T_prob& theta) {
      static const std::string function("stan::prob::bernoulli_cdf");
      
      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero argument lenghts
      if (!(stan::length(n) && stan::length(theta)))
        return 1.0;
          
      double P(1.0);
          
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
        if (value_of(n_vec[i]) >= 1) continue;
        else {
          const double Pi = 1 - value_of(theta_vec[i]);
                    
          P *= Pi;
                    
          if (!is_constant_struct<T_prob>::value)
            operands_and_partials.d_x1[i] += - 1 / Pi;
        }
      }
          
      if (!is_constant_struct<T_prob>::value) {
        for(size_t i = 0; i < stan::length(theta); ++i) operands_and_partials.d_x1[i] *= P;
      }
      return operands_and_partials.to_var(P);
    }
      
    template <typename T_n, typename T_prob>
    typename return_type<T_prob>::type
    bernoulli_cdf_log(const T_n& n, const T_prob& theta) {
      static const std::string function("stan::prob::bernoulli_cdf_log");
      
      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero argument lenghts
      if (!(stan::length(n) && stan::length(theta)))
        return 0.0;
          
      double P(0.0);
          
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
          
      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
      agrad::OperandsAndPartials<T_prob> operands_and_partials(theta);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(n); i++) {
        if (value_of(n_vec[i]) < 0) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
          
      for (size_t i = 0; i < size; i++) {
          
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(n_vec[i]) >= 1) continue;
        else {
          const double Pi = 1 - value_of(theta_vec[i]);
                    
          P += log(Pi);
                    
          if (!is_constant_struct<T_prob>::value)
            operands_and_partials.d_x1[i] -= 1 / Pi;
        }
      }
       
      return operands_and_partials.to_var(P);
    }

    template <typename T_n, typename T_prob>
    typename return_type<T_prob>::type
    bernoulli_ccdf_log(const T_n& n, const T_prob& theta) {
      static const std::string function("stan::prob::bernoulli_ccdf_log");
      
      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;
          
      // Ensure non-zero argument lenghts
      if (!(stan::length(n) && stan::length(theta)))
        return 0.0;
          
      double P(0.0);
          
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
          
      // Compute vectorized cdf_log and gradient
      using stan::math::value_of;
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
        if (value_of(n_vec[i]) >= 1) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
        else {
          const double Pi = value_of(theta_vec[i]);
                    
          P += log(Pi);
                    
          if (!is_constant_struct<T_prob>::value)
            operands_and_partials.d_x1[i] += 1 / Pi;
        }
      }
       
      return operands_and_partials.to_var(P);
    }


    template <class RNG>
    inline int
    bernoulli_rng(const double theta,
                  RNG& rng) {
      using boost::variate_generator;
      using boost::bernoulli_distribution;

      static const std::string function("stan::prob::bernoulli_rng");

      using stan::error_handling::check_finite;
      using stan::error_handling::check_bounded;
 
      check_finite(function, "Probability parameter", theta);
      check_bounded(function, "Probability parameter", theta, 0, 1);

      variate_generator<RNG&, bernoulli_distribution<> >
        bernoulli_rng(rng, bernoulli_distribution<>(theta));
      return bernoulli_rng();
    }
  }
}
#endif
