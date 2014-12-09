#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_2_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__NEG_BINOMIAL_2_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_less.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/internal_math.hpp>
#include <stan/prob/distributions/univariate/continuous/beta.hpp>
#include <stan/prob/distributions/univariate/continuous/gamma.hpp>
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

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

      static const std::string function("stan::prob::neg_binomial_2_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(mu)
            && stan::length(phi)))
        return 0.0;

      double logp(0.0);
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
      using stan::math::log_sum_exp;
      using stan::math::binomial_coefficient_log;
      using boost::math::digamma;
      using boost::math::lgamma;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_location> mu_vec(mu);
      VectorView<const T_precision> phi_vec(phi);
      size_t size = max_size(n, mu, phi);

      agrad::OperandsAndPartials<T_location, T_precision>
        operands_and_partials(mu, phi);

      size_t len_ep = max_size(mu, phi);
      size_t len_np = max_size(n, phi);
      
      DoubleVectorView<true, is_vector<T_location>::value>
        mu__(length(mu));
      for (size_t i = 0, size = length(mu); i < size; ++i)
        mu__[i] = value_of(mu_vec[i]);
  
      DoubleVectorView<true, is_vector<T_precision>::value>
        phi__(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        phi__[i] = value_of(phi_vec[i]);
      
      DoubleVectorView<true, is_vector<T_precision>::value>
        log_phi(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        log_phi[i] = log(phi__[i]);

      DoubleVectorView<true, (is_vector<T_location>::value
                             || is_vector<T_precision>::value)>
        log_mu_plus_phi(len_ep);
      for (size_t i = 0; i < len_ep; ++i)
        log_mu_plus_phi[i] = log(mu__[i] + phi__[i]);

      DoubleVectorView<true, (is_vector<T_n>::value
                             || is_vector<T_precision>::value)>
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
            + log_phi[i] - log_mu_plus_phi[i] - digamma(phi__[i]) + digamma(n_plus_phi[i]);
      }
      return operands_and_partials.to_var(logp);
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


    // NegBinomial(n|eta,phi)  [phi > 0;  n >= 0]
    template <bool propto,
              typename T_n,
              typename T_log_location, typename T_precision>
    typename return_type<T_log_location, T_precision>::type
    neg_binomial_2_log_log(const T_n& n,
                     const T_log_location& eta,
                     const T_precision& phi) {

      static const std::string function("stan::prob::neg_binomial_log");

      using stan::error_handling::check_finite;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_positive_finite;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(eta)
            && stan::length(phi)))
        return 0.0;

      double logp(0.0);
      check_nonnegative(function, "Failures variable", n);
      check_finite(function, "Log location parameter", eta);
      check_positive_finite(function, "Precision parameter", phi);
      check_consistent_sizes(function,
                             "Failures variable", n,
                             "Log location parameter", eta,
                             "Precision parameter", phi);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_log_location,T_precision>::value)
        return 0.0;

      using stan::math::multiply_log;
      using stan::math::log_sum_exp;
      using stan::math::binomial_coefficient_log;
      using boost::math::digamma;
      using boost::math::lgamma;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_log_location> eta_vec(eta);
      VectorView<const T_precision> phi_vec(phi);
      size_t size = max_size(n, eta, phi);

      agrad::OperandsAndPartials<T_log_location,T_precision>
        operands_and_partials(eta,phi);

      size_t len_ep = max_size(eta, phi);
      size_t len_np = max_size(n, phi);

      DoubleVectorView<true, is_vector<T_log_location>::value>
        eta__(length(eta));
      for (size_t i = 0, size = length(eta); i < size; ++i)
        eta__[i] = value_of(eta_vec[i]);
  
      DoubleVectorView<true, is_vector<T_precision>::value>
        phi__(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        phi__[i] = value_of(phi_vec[i]);  
        

      DoubleVectorView<true, is_vector<T_precision>::value>
        log_phi(length(phi));
      for (size_t i = 0, size = length(phi); i < size; ++i)
        log_phi[i] = log(phi__[i]);

      DoubleVectorView<true, (is_vector<T_log_location>::value
                             || is_vector<T_precision>::value)>
        logsumexp_eta_logphi(len_ep);
      for (size_t i = 0; i < len_ep; ++i)
        logsumexp_eta_logphi[i] = log_sum_exp(eta__[i], log_phi[i]);

      DoubleVectorView<true, (is_vector<T_n>::value
                             || is_vector<T_precision>::value)>
        n_plus_phi(len_np);
      for (size_t i = 0; i < len_np; ++i)
        n_plus_phi[i] = n_vec[i] + phi__[i];

      for (size_t i = 0; i < size; i++) {
        if (include_summand<propto>::value)
          logp -= lgamma(n_vec[i] + 1.0);
        if (include_summand<propto,T_precision>::value)
          logp += multiply_log(phi__[i], phi__[i]) - lgamma(phi__[i]);
        if (include_summand<propto,T_log_location,T_precision>::value)
          logp -= (n_plus_phi[i])*logsumexp_eta_logphi[i];
        if (include_summand<propto,T_log_location>::value)
          logp += n_vec[i]*eta__[i];
        if (include_summand<propto,T_precision>::value)
          logp += lgamma(n_plus_phi[i]);

        if (!is_constant_struct<T_log_location>::value)
          operands_and_partials.d_x1[i]
            += n_vec[i] - n_plus_phi[i]
            / (phi__[i]/exp(eta__[i]) + 1.0);
        if (!is_constant_struct<T_precision>::value)
          operands_and_partials.d_x2[i]
            += 1.0 - n_plus_phi[i]/(exp(eta__[i]) + phi__[i])
            + log_phi[i] - logsumexp_eta_logphi[i] - digamma(phi__[i]) + digamma(n_plus_phi[i]);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_n,
              typename T_log_location, typename T_precision>
    inline
    typename return_type<T_log_location, T_precision>::type
    neg_binomial_2_log_log(const T_n& n,
                     const T_log_location& eta,
                     const T_precision& phi) {
      return neg_binomial_2_log_log<false>(n,eta,phi);
    }
    
    template <typename T_n, typename T_location, 
              typename T_precision>
    typename return_type<T_location, T_precision>::type
    neg_binomial_2_cdf(const T_n& n,
                       const T_location& mu,
                       const T_precision& phi) {
                         
      // Size checks
      if ( !( stan::length(n) && stan::length(mu) 
              && stan::length(phi) ) ) 
        return 1.0;
        
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_less;
      
      static const std::string function("stan::prob::neg_binomial_2_cdf");
      check_positive_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Precision parameter", phi);
      check_not_nan(function, "Random variable", n);
      check_consistent_sizes(function, 
                             "Random variable", n, 
                             "Location parameter", mu, 
                             "Precision Parameter", phi);
      
      VectorView<const T_n> n_vec(n);
      VectorView<const T_location> mu_vec(mu);
      VectorView<const T_precision> phi_vec(phi);
      
      size_t size_phi_mu = max_size(mu, phi);
      size_t size_n = length(n);
      
      std::vector<typename return_type<T_location, T_precision>::type> phi_mu(size_phi_mu);
      std::vector<typename return_type<T_n>::type> np1(size_n);

      for (size_t i = 0; i < size_phi_mu; i++)
        phi_mu[i] = phi_vec[i]/(phi_vec[i]+mu_vec[i]);

      for (size_t i = 0; i < size_n; i++)
        if (n_vec[i] < 0)
          return 0.0;         
        else
          np1[i] = n_vec[i] + 1.0;
      
      if (size_n == 1) {
        if (size_phi_mu == 1)
          return beta_cdf(phi_mu[0], phi, np1[0]);                       
        else
          return beta_cdf(phi_mu, phi, np1[0]);                                 
      }
      else {
        if (size_phi_mu == 1)
          return beta_cdf(phi_mu[0], phi, np1);                       
        else
          return beta_cdf(phi_mu, phi, np1);                                 
      }
    }             
    
    template <typename T_n, typename T_location, 
              typename T_precision>
    typename return_type<T_location, T_precision>::type
    neg_binomial_2_cdf_log(const T_n& n,
                       const T_location& mu,
                       const T_precision& phi) {
                         
      // Size checks
      if ( !( stan::length(n) && stan::length(mu) 
              && stan::length(phi) ) ) 
        return 0.0;
        
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_less;
      
      static const std::string function("stan::prob::neg_binomial_2_cdf");
      check_positive_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Precision parameter", phi);
      check_not_nan(function, "Random variable", n);
      check_consistent_sizes(function, 
                             "Random variable", n, 
                             "Location parameter", mu, 
                             "Precision Parameter", phi);
      
      VectorView<const T_n> n_vec(n);
      VectorView<const T_location> mu_vec(mu);
      VectorView<const T_precision> phi_vec(phi);
      
      size_t size_phi_mu = max_size(mu, phi);
      size_t size_n = length(n);
      
      std::vector<typename return_type<T_location, T_precision>::type> phi_mu(size_phi_mu);
      std::vector<typename return_type<T_n>::type> np1(size_n);

      for (size_t i = 0; i < size_phi_mu; i++)
        phi_mu[i] = phi_vec[i]/(phi_vec[i]+mu_vec[i]);

      for (size_t i = 0; i < size_n; i++)
        if (n_vec[i] < 0)
          return log(0.0);
        else
          np1[i] = n_vec[i] + 1.0;
      
      if (size_n == 1) {
        if (size_phi_mu == 1)
          return beta_cdf_log(phi_mu[0], phi, np1[0]);                       
        else
          return beta_cdf_log(phi_mu, phi, np1[0]);                                 
      }
      else {
        if (size_phi_mu == 1)
          return beta_cdf_log(phi_mu[0], phi, np1);                       
        else
          return beta_cdf_log(phi_mu, phi, np1);                                 
      }
    }             

    namespace {
      
      //modified version of beta_ccdf_log
      //used in neg_binomial_2_ccdf_log
      //modifications made: remove unnecessary validations and include:
      //if (beta_dbl < 1)
      //  continue;
      template <typename T_y, typename T_scale_succ, typename T_scale_fail>
      typename return_type<T_y,T_scale_succ,T_scale_fail>::type
      beta_ccdf_log_modified(const T_y& y, const T_scale_succ& alpha, 
                    const T_scale_fail& beta) {

        using stan::math::value_of;

        double ccdf_log(0.0);
        
        // Wrap arguments in vectors
        VectorView<const T_y> y_vec(y);
        VectorView<const T_scale_succ> alpha_vec(alpha);
        VectorView<const T_scale_fail> beta_vec(beta);
        size_t N = max_size(y, alpha, beta);

        agrad::OperandsAndPartials<T_y, T_scale_succ, T_scale_fail> 
          operands_and_partials(y, alpha, beta);

        // Compute CDF and its gradients
        using boost::math::ibeta;
        using boost::math::ibeta_derivative;
        using boost::math::digamma;
          
        // Cache a few expensive function calls if alpha or beta is a parameter
        DoubleVectorView<!is_constant_struct<T_scale_succ>::value 
                         || !is_constant_struct<T_scale_fail>::value,
          is_vector<T_scale_succ>::value || is_vector<T_scale_fail>::value>
          digamma_alpha_vec(max_size(alpha, beta));
        DoubleVectorView<!is_constant_struct<T_scale_succ>::value 
                         || !is_constant_struct<T_scale_fail>::value,
          is_vector<T_scale_succ>::value || is_vector<T_scale_fail>::value>
          digamma_beta_vec(max_size(alpha, beta));
        DoubleVectorView<!is_constant_struct<T_scale_succ>::value
                         || !is_constant_struct<T_scale_fail>::value,
          is_vector<T_scale_succ>::value || is_vector<T_scale_fail>::value>
          digamma_sum_vec(max_size(alpha, beta));        
        DoubleVectorView<!is_constant_struct<T_scale_succ>::value
                         || !is_constant_struct<T_scale_fail>::value,
          is_vector<T_scale_succ>::value || is_vector<T_scale_fail>::value>
          betafunc_vec(max_size(alpha, beta));
          
        if (!is_constant_struct<T_scale_succ>::value 
            || !is_constant_struct<T_scale_fail>::value) {
              
          for (size_t i = 0; i < N; i++) {

            const double alpha_dbl = value_of(alpha_vec[i]);
            const double beta_dbl = value_of(beta_vec[i]);
                  
            digamma_alpha_vec[i] = digamma(alpha_dbl);
            digamma_beta_vec[i] = digamma(beta_dbl);
            digamma_sum_vec[i] = digamma(alpha_dbl + beta_dbl);
            betafunc_vec[i] = boost::math::beta(alpha_dbl, beta_dbl);
          }
        }
          
        // Compute vectorized CDFLog and gradient
        for (size_t n = 0; n < N; n++) {
                
          // Pull out values
          const double y_dbl = value_of(y_vec[n]);
          const double alpha_dbl = value_of(alpha_vec[n]);
          const double beta_dbl = value_of(beta_vec[n]);
          
          if (beta_dbl < 1)
            continue;
                    
          // Compute
          const double Pn = 1.0 - ibeta(alpha_dbl, beta_dbl, y_dbl);

          ccdf_log += log(Pn);
                    
          if (!is_constant_struct<T_y>::value)
            operands_and_partials.d_x1[n] -= 
              ibeta_derivative(alpha_dbl, beta_dbl, y_dbl) / Pn;

          double g1 = 0;
          double g2 = 0;
                
          if (!is_constant_struct<T_scale_succ>::value
              || !is_constant_struct<T_scale_fail>::value) {
            stan::math::gradRegIncBeta(g1, g2, alpha_dbl, beta_dbl, y_dbl, 
                                       digamma_alpha_vec[n], 
                                       digamma_beta_vec[n], digamma_sum_vec[n], 
                                       betafunc_vec[n]);
          }
          if (!is_constant_struct<T_scale_succ>::value)
            operands_and_partials.d_x2[n] -= g1 / Pn;
          if (!is_constant_struct<T_scale_fail>::value)
            operands_and_partials.d_x3[n] -= g2 / Pn;
        }
          
        return operands_and_partials.to_var(ccdf_log);
      }      
      
    }
    
    template <typename T_n, typename T_location, 
              typename T_precision>
    typename return_type<T_location, T_precision>::type
    neg_binomial_2_ccdf_log(const T_n& n,
                       const T_location& mu,
                       const T_precision& phi) {
                         
      // Size checks
      if ( !( stan::length(n) && stan::length(mu) 
              && stan::length(phi) ) ) 
        return 0.0;
        
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_less;
      
      static const std::string function("stan::prob::neg_binomial_2_cdf");
      check_positive_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Precision parameter", phi);
      check_not_nan(function, "Random variable", n);
      check_consistent_sizes(function, 
                             "Random variable", n, 
                             "Location parameter", mu, 
                             "Precision Parameter", phi);
      
      VectorView<const T_n> n_vec(n);
      VectorView<const T_location> mu_vec(mu);
      VectorView<const T_precision> phi_vec(phi);
      
      size_t size_phi_mu = max_size(mu, phi);
      size_t size_n = length(n);
      
      std::vector<typename return_type<T_location, T_precision>::type> phi_mu(size_phi_mu);
      std::vector<typename return_type<T_n>::type> np1(size_n);
      std::vector<size_t> np_zeros;

      for (size_t i = 0; i < size_phi_mu; i++)
        phi_mu[i] = phi_vec[i]/(phi_vec[i]+mu_vec[i]);

      for (size_t i = 0; i < size_n; i++)
        if (n_vec[i] < 0)
          np1[i] = 0.9;        
        else
          np1[i] = n_vec[i] + 1.0;
              
      if (size_n == 1) {
        if (size_phi_mu == 1)
          return beta_ccdf_log_modified(phi_mu[0], phi, np1[0]);                       
        else
          return beta_ccdf_log_modified(phi_mu, phi, np1[0]);                   
      }
      else {
        if (size_phi_mu == 1)
          return beta_ccdf_log_modified(phi_mu[0], phi, np1);                       
        else
          return beta_ccdf_log_modified(phi_mu, phi, np1);                                 
      }
    
    }             
    
    template <class RNG>
    inline int
    neg_binomial_2_rng(const double mu,
                     const double phi,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;

      static const std::string function("stan::prob::neg_binomial_2_rng");

      using stan::error_handling::check_positive_finite;

      check_positive_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Precision parameter", phi);
                            

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

      static const std::string function("stan::prob::neg_binomial_2_log_rng");

      using stan::error_handling::check_finite;
      using stan::error_handling::check_positive_finite;

      check_finite(function, "Log-location parameter", eta);
      check_positive_finite(function, "Precision parameter", phi);


      return stan::prob::poisson_rng(stan::prob::gamma_rng(phi,phi/std::exp(eta),
                                                           rng),rng);
    }
  }
}
#endif
