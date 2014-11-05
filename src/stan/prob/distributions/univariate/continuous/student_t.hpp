#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__STUDENT_T_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__STUDENT_T_HPP

#include <boost/random/student_t_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/square.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/internal_math.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of the Student-t density for the given y, nu, mean, and
     * scale parameter.  The scale parameter must be greater
     * than 0.
     *
     * \f{eqnarray*}{
     y &\sim& t_{\nu} (\mu, \sigma^2) \\
     \log (p (y \,|\, \nu, \mu, \sigma) ) &=& \log \left( \frac{\Gamma((\nu + 1) /2)}
     {\Gamma(\nu/2)\sqrt{\nu \pi} \sigma} \left( 1 + \frac{1}{\nu} (\frac{y - \mu}{\sigma})^2 \right)^{-(\nu + 1)/2} \right) \\
     &=& \log( \Gamma( (\nu+1)/2 )) - \log (\Gamma (\nu/2) - \frac{1}{2} \log(\nu \pi) - \log(\sigma)
     -\frac{\nu + 1}{2} \log (1 + \frac{1}{\nu} (\frac{y - \mu}{\sigma})^2)
     \f}
     * 
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param mu The mean of the Student-t distribution.
     * @param sigma The scale parameter of the Student-t distribution.
     * @return The log of the Student-t density at y.
     * @throw std::domain_error if sigma is not greater than 0.
     * @throw std::domain_error if nu is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_loc Type of location.
     * @tparam T_scale Type of scale.
     */
    template <bool propto, typename T_y, typename T_dof, 
              typename T_loc, typename T_scale>
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
      static const std::string function("stan::prob::student_t_log");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(nu) 
            && stan::length(mu)
            && stan::length(sigma)))
        return 0.0;

      double logp(0.0);

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu,
                             "Location parameter", mu,
                             "Scale parameter", sigma);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, nu, mu, sigma);

      using std::log;
      using boost::math::digamma;
      using boost::math::lgamma;
      using stan::math::square;
      using stan::math::value_of;

      DoubleVectorView<include_summand<propto,T_y,T_dof,T_loc,T_scale>::value,
        is_vector<T_dof>::value> half_nu(length(nu));
      for (size_t i = 0; i < length(nu); i++) 
        if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value) 
          half_nu[i] = 0.5 * value_of(nu_vec[i]);
      DoubleVectorView<include_summand<propto,T_dof>::value,
        is_vector<T_dof>::value> lgamma_half_nu(length(nu));
      DoubleVectorView<include_summand<propto,T_dof>::value,
        is_vector<T_dof>::value> lgamma_half_nu_plus_half(length(nu));
      if (include_summand<propto,T_dof>::value)
        for (size_t i = 0; i < length(nu); i++) {
          lgamma_half_nu[i] = lgamma(half_nu[i]);
          lgamma_half_nu_plus_half[i] = lgamma(half_nu[i] + 0.5);
        }
      DoubleVectorView<!is_constant_struct<T_dof>::value,
        is_vector<T_dof>::value> digamma_half_nu(length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
        is_vector<T_dof>::value> digamma_half_nu_plus_half(length(nu));
      if (!is_constant_struct<T_dof>::value)
        for (size_t i = 0; i < length(nu); i++) {
          digamma_half_nu[i] = digamma(half_nu[i]);
          digamma_half_nu_plus_half[i] = digamma(half_nu[i] + 0.5);
        }
    


      DoubleVectorView<include_summand<propto,T_dof>::value,
        is_vector<T_dof>::value> log_nu(length(nu));
      for (size_t i = 0; i < length(nu); i++)
        if (include_summand<propto,T_dof>::value)
          log_nu[i] = log(value_of(nu_vec[i]));
      DoubleVectorView<include_summand<propto,T_scale>::value,
        is_vector<T_scale>::value> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++)
        if (include_summand<propto,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));

      DoubleVectorView<include_summand<propto,T_y,T_dof,T_loc,T_scale>::value,
        is_vector<T_y>::value 
                       || is_vector<T_dof>::value 
                       || is_vector<T_loc>::value 
                       || is_vector<T_scale>::value> 
        square_y_minus_mu_over_sigma__over_nu(N);

      DoubleVectorView<include_summand<propto,T_y,T_dof,T_loc,T_scale>::value,
        is_vector<T_y>::value 
                       || is_vector<T_dof>::value
                       || is_vector<T_loc>::value 
                       || is_vector<T_scale>::value> 
        log1p_exp(N);

      for (size_t i = 0; i < N; i++) 
        if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value) {
          const double y_dbl = value_of(y_vec[i]);
          const double mu_dbl = value_of(mu_vec[i]);
          const double sigma_dbl = value_of(sigma_vec[i]);
          const double nu_dbl = value_of(nu_vec[i]);
          square_y_minus_mu_over_sigma__over_nu[i] 
            = square((y_dbl - mu_dbl) / sigma_dbl) / nu_dbl;
          log1p_exp[i] = log1p(square_y_minus_mu_over_sigma__over_nu[i]);
        }

      agrad::OperandsAndPartials<T_y,T_dof,T_loc,T_scale>
        operands_and_partials(y,nu,mu,sigma);
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double nu_dbl = value_of(nu_vec[n]);
        if (include_summand<propto>::value)
          logp += NEG_LOG_SQRT_PI;
        if (include_summand<propto,T_dof>::value)
          logp += lgamma_half_nu_plus_half[n] - lgamma_half_nu[n]
            -  0.5 * log_nu[n];
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
          logp -= (half_nu[n] + 0.5)
            * log1p_exp[n];
  
        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n] 
            += -(half_nu[n]+0.5)
            * 1.0 / (1.0 + square_y_minus_mu_over_sigma__over_nu[n])
            * (2.0 * (y_dbl - mu_dbl) / square(sigma_dbl) / nu_dbl);
        }
        if (!is_constant_struct<T_dof>::value) {
          const double inv_nu = 1.0 / nu_dbl;
          operands_and_partials.d_x2[n] 
            += 0.5*digamma_half_nu_plus_half[n] - 0.5*digamma_half_nu[n]
            - 0.5 * inv_nu
            - 0.5*log1p_exp[n]
            + (half_nu[n] + 0.5)
            * (1.0/(1.0 + square_y_minus_mu_over_sigma__over_nu[n])
               * square_y_minus_mu_over_sigma__over_nu[n] * inv_nu);
        }
        if (!is_constant_struct<T_loc>::value) {
          operands_and_partials.d_x3[n] 
            -= (half_nu[n] + 0.5) 
            / (1.0 + square_y_minus_mu_over_sigma__over_nu[n]) 
            * (2.0 * (mu_dbl - y_dbl) / (sigma_dbl*sigma_dbl*nu_dbl));
        }
        if (!is_constant_struct<T_scale>::value) {
          const double inv_sigma = 1.0 / sigma_dbl;
          operands_and_partials.d_x4[n] 
            += -inv_sigma
            + (nu_dbl + 1.0) / (1.0 + square_y_minus_mu_over_sigma__over_nu[n]) 
            * (square_y_minus_mu_over_sigma__over_nu[n] * inv_sigma);
        }
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
      return student_t_log<false>(y,nu,mu,sigma);
    }
      
    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    typename return_type<T_y, T_dof, T_loc, T_scale>::type
    student_t_cdf(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
          
      // Size checks
      if (!(stan::length(y) && stan::length(nu) && stan::length(mu) 
            && stan::length(sigma))) 
        return 1.0;
      
      static const std::string function("stan::prob::student_t_cdf");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
          
      double P(1.0);
          
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, nu, mu, sigma);
          
      agrad::OperandsAndPartials<T_y, T_dof, T_loc, T_scale> 
        operands_and_partials(y, nu, mu, sigma);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == -std::numeric_limits<double>::infinity()) 
          return operands_and_partials.to_var(0.0);
      }
          
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;
      using boost::math::digamma;
      using boost::math::beta;
          
      // Cache a few expensive function calls if nu is a parameter
      double digammaHalf = 0;
          
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        digamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        digammaNu_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value>
        digammaNuPlusHalf_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        betaNuHalf_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        digammaHalf = digamma(0.5);
              
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double nu_dbl = value_of(nu_vec[i]);
                  
          digammaNu_vec[i] = digamma(0.5 * nu_dbl);
          digammaNuPlusHalf_vec[i] = digamma(0.5 + 0.5 * nu_dbl);
          betaNuHalf_vec[i] = beta(0.5, 0.5 * nu_dbl);
        }
      }
          
      // Compute vectorized CDF and gradient
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }
                    
        const double sigma_inv = 1.0 / value_of(sigma_vec[n]);
        const double t = (value_of(y_vec[n]) - value_of(mu_vec[n])) * sigma_inv;
        const double nu_dbl = value_of(nu_vec[n]);
        const double q = nu_dbl / (t * t);
        const double r = 1.0 / (1.0 + q);
        const double J = 2 * r * r * q / t;
        double zJacobian = t > 0 ? - 0.5 : 0.5;
                    
        if(q < 2)
          {

            double z = ibeta(0.5 * nu_dbl, 0.5, 1.0 - r);
            const double Pn = t > 0 ? 1.0 - 0.5 * z : 0.5 * z;
            const double d_ibeta = ibeta_derivative(0.5 * nu_dbl, 0.5, 1.0 - r);
                      
            P *= Pn;

            if (!is_constant_struct<T_y>::value)
              operands_and_partials.d_x1[n] 
                += - zJacobian * d_ibeta * J * sigma_inv / Pn;
            if (!is_constant_struct<T_dof>::value) {
                          
              double g1 = 0;
              double g2 = 0;
                          
              stan::math::gradRegIncBeta(g1, g2, 0.5 * nu_dbl, 0.5, 1.0 - r, 
                                         digammaNu_vec[n], digammaHalf,
                                         digammaNuPlusHalf_vec[n], 
                                         betaNuHalf_vec[n]);
                          
              operands_and_partials.d_x2[n] 
                += zJacobian * ( d_ibeta * (r / t) * (r / t) + 0.5 * g1 ) / Pn;
            }

            if (!is_constant_struct<T_loc>::value)
              operands_and_partials.d_x3[n] 
                += zJacobian * d_ibeta * J * sigma_inv / Pn;
            if (!is_constant_struct<T_scale>::value)
              operands_and_partials.d_x4[n] 
                += zJacobian * d_ibeta * J * sigma_inv * t / Pn;
                      
          }
        else {
                  
            double z = 1 - ibeta(0.5, 0.5 * nu_dbl, r);
            zJacobian *= -1;
                  
            const double Pn = t > 0 ? 1.0 - 0.5 * z : 0.5 * z;
                  
            double d_ibeta = ibeta_derivative(0.5, 0.5 * nu_dbl, r);
                  
            P *= Pn;
                  
            if (!is_constant_struct<T_y>::value)
              operands_and_partials.d_x1[n] 
                += zJacobian * d_ibeta * J * sigma_inv / Pn;
            if (!is_constant_struct<T_dof>::value) {
                      
              double g1 = 0;
              double g2 = 0;
                      
              stan::math::gradRegIncBeta(g1, g2, 0.5, 0.5 * nu_dbl, r, 
                                         digammaHalf, digammaNu_vec[n], 
                                         digammaNuPlusHalf_vec[n], 
                                         betaNuHalf_vec[n]);
                      
              operands_and_partials.d_x2[n] 
                += zJacobian * ( - d_ibeta * (r / t) * (r / t) + 0.5 * g2 ) / Pn;
            }
            if (!is_constant_struct<T_loc>::value)
              operands_and_partials.d_x3[n] 
                += - zJacobian * d_ibeta * J * sigma_inv / Pn;
            if (!is_constant_struct<T_scale>::value)
              operands_and_partials.d_x4[n] 
                += - zJacobian * d_ibeta * J * sigma_inv * t / Pn;
          }
      }
          
      if (!is_constant_struct<T_y>::value)
        for(size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= P;
      if (!is_constant_struct<T_dof>::value)
        for(size_t n = 0; n < stan::length(nu); ++n)
          operands_and_partials.d_x2[n] *= P;
      if (!is_constant_struct<T_loc>::value)
        for(size_t n = 0; n < stan::length(mu); ++n) 
          operands_and_partials.d_x3[n] *= P;
      if (!is_constant_struct<T_scale>::value)
        for(size_t n = 0; n < stan::length(sigma); ++n)
          operands_and_partials.d_x4[n] *= P;
          
      return operands_and_partials.to_var(P);
    }
          
    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    typename return_type<T_y, T_dof, T_loc, T_scale>::type
    student_t_cdf_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                      const T_scale& sigma) {
          
      // Size checks
      if (!(stan::length(y) && stan::length(nu) && stan::length(mu) 
            && stan::length(sigma))) 
        return 0.0;
      
      static const std::string function("stan::prob::student_t_cdf_log");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
          
      double P(0.0);
          
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, nu, mu, sigma);
          
      agrad::OperandsAndPartials<T_y, T_dof, T_loc, T_scale> 
        operands_and_partials(y, nu, mu, sigma);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == -std::numeric_limits<double>::infinity()) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
          
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;
      using boost::math::digamma;
      using boost::math::beta;
          
      // Cache a few expensive function calls if nu is a parameter
      double digammaHalf = 0;
          
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        digamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        digammaNu_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value>
        digammaNuPlusHalf_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        betaNuHalf_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        digammaHalf = digamma(0.5);
              
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double nu_dbl = value_of(nu_vec[i]);
                  
          digammaNu_vec[i] = digamma(0.5 * nu_dbl);
          digammaNuPlusHalf_vec[i] = digamma(0.5 + 0.5 * nu_dbl);
          betaNuHalf_vec[i] = beta(0.5, 0.5 * nu_dbl);
        }
      }
          
      // Compute vectorized cdf_log and gradient
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }
                    
        const double sigma_inv = 1.0 / value_of(sigma_vec[n]);
        const double t = (value_of(y_vec[n]) - value_of(mu_vec[n])) * sigma_inv;
        const double nu_dbl = value_of(nu_vec[n]);
        const double q = nu_dbl / (t * t);
        const double r = 1.0 / (1.0 + q);
        const double J = 2 * r * r * q / t;
        double zJacobian = t > 0 ? - 0.5 : 0.5;
                    
        if(q < 2) {
            double z = ibeta(0.5 * nu_dbl, 0.5, 1.0 - r);
            const double Pn = t > 0 ? 1.0 - 0.5 * z : 0.5 * z;
            const double d_ibeta = ibeta_derivative(0.5 * nu_dbl, 0.5, 1.0 - r);
                      
            P += log(Pn);

            if (!is_constant_struct<T_y>::value)
              operands_and_partials.d_x1[n] 
                += - zJacobian * d_ibeta * J * sigma_inv / Pn;
                      
            if (!is_constant_struct<T_dof>::value) {
                          
              double g1 = 0;
              double g2 = 0;
                          
              stan::math::gradRegIncBeta(g1, g2, 0.5 * nu_dbl, 0.5, 1.0 - r, 
                                         digammaNu_vec[n], digammaHalf,
                                         digammaNuPlusHalf_vec[n], 
                                         betaNuHalf_vec[n]);
                          
              operands_and_partials.d_x2[n] 
                += zJacobian * ( d_ibeta * (r / t) * (r / t) + 0.5 * g1 ) / Pn;
            }

            if (!is_constant_struct<T_loc>::value)
              operands_and_partials.d_x3[n] 
                += zJacobian * d_ibeta * J * sigma_inv / Pn;
            if (!is_constant_struct<T_scale>::value)
              operands_and_partials.d_x4[n] 
                += zJacobian * d_ibeta * J * sigma_inv * t / Pn;
                      
          }
        else {
                  
            double z = 1 - ibeta(0.5, 0.5 * nu_dbl, r);
            zJacobian *= -1;
                  
            const double Pn = t > 0 ? 1.0 - 0.5 * z : 0.5 * z;
                  
            double d_ibeta = ibeta_derivative(0.5, 0.5 * nu_dbl, r);
                  
            P += log(Pn);
                  
            if (!is_constant_struct<T_y>::value)
              operands_and_partials.d_x1[n] 
                += zJacobian * d_ibeta * J * sigma_inv / Pn;
                  
            if (!is_constant_struct<T_dof>::value) {
                      
              double g1 = 0;
              double g2 = 0;
                      
              stan::math::gradRegIncBeta(g1, g2, 0.5, 0.5 * nu_dbl, r, 
                                         digammaHalf, digammaNu_vec[n], 
                                         digammaNuPlusHalf_vec[n], 
                                         betaNuHalf_vec[n]);
                      
              operands_and_partials.d_x2[n] 
                += zJacobian * ( - d_ibeta * (r / t) * (r / t) + 0.5 * g2 ) / Pn;
            }
                  
            if (!is_constant_struct<T_loc>::value)
              operands_and_partials.d_x3[n] 
                += - zJacobian * d_ibeta * J * sigma_inv / Pn;
            if (!is_constant_struct<T_scale>::value)
              operands_and_partials.d_x4[n] 
                += - zJacobian * d_ibeta * J * sigma_inv * t / Pn;
          }
      }

      return operands_and_partials.to_var(P);
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    typename return_type<T_y, T_dof, T_loc, T_scale>::type
    student_t_ccdf_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                      const T_scale& sigma) {
          
      // Size checks
      if (!(stan::length(y) && stan::length(nu) && stan::length(mu) 
            && stan::length(sigma))) 
        return 0.0;
      
      static const std::string function("stan::prob::student_t_ccdf_log");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;
          
      double P(0.0);
          
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, nu, mu, sigma);
          
      agrad::OperandsAndPartials<T_y, T_dof, T_loc, T_scale> 
        operands_and_partials(y, nu, mu, sigma);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == -std::numeric_limits<double>::infinity()) 
          return operands_and_partials.to_var(0.0);
      }
          
      using boost::math::ibeta;
      using boost::math::ibeta_derivative;
      using boost::math::digamma;
      using boost::math::beta;
          
      // Cache a few expensive function calls if nu is a parameter
      double digammaHalf = 0;
          
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        digamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        digammaNu_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value>
        digammaNuPlusHalf_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> 
        betaNuHalf_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        digammaHalf = digamma(0.5);
              
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double nu_dbl = value_of(nu_vec[i]);
                  
          digammaNu_vec[i] = digamma(0.5 * nu_dbl);
          digammaNuPlusHalf_vec[i] = digamma(0.5 + 0.5 * nu_dbl);
          betaNuHalf_vec[i] = beta(0.5, 0.5 * nu_dbl);
        }
      }
          
      // Compute vectorized cdf_log and gradient
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          return operands_and_partials.to_var(stan::math::negative_infinity());
        }
                    
        const double sigma_inv = 1.0 / value_of(sigma_vec[n]);
        const double t = (value_of(y_vec[n]) - value_of(mu_vec[n])) * sigma_inv;
        const double nu_dbl = value_of(nu_vec[n]);
        const double q = nu_dbl / (t * t);
        const double r = 1.0 / (1.0 + q);
        const double J = 2 * r * r * q / t;
        double zJacobian = t > 0 ? - 0.5 : 0.5;
                    
        if(q < 2) {
            double z = ibeta(0.5 * nu_dbl, 0.5, 1.0 - r);
            const double Pn = t > 0 ? 0.5 * z : 1.0 - 0.5 * z;
            const double d_ibeta = ibeta_derivative(0.5 * nu_dbl, 0.5, 1.0 - r);
                      
            P += log(Pn);

            if (!is_constant_struct<T_y>::value)
              operands_and_partials.d_x1[n] 
                += zJacobian * d_ibeta * J * sigma_inv / Pn;
                      
            if (!is_constant_struct<T_dof>::value) {
                          
              double g1 = 0;
              double g2 = 0;
                          
              stan::math::gradRegIncBeta(g1, g2, 0.5 * nu_dbl, 0.5, 1.0 - r, 
                                         digammaNu_vec[n], digammaHalf,
                                         digammaNuPlusHalf_vec[n], 
                                         betaNuHalf_vec[n]);
                          
              operands_and_partials.d_x2[n] 
                -= zJacobian * ( d_ibeta * (r / t) * (r / t) + 0.5 * g1 ) / Pn;
            }

            if (!is_constant_struct<T_loc>::value)
              operands_and_partials.d_x3[n] 
                -= zJacobian * d_ibeta * J * sigma_inv / Pn;
            if (!is_constant_struct<T_scale>::value)
              operands_and_partials.d_x4[n] 
                -= zJacobian * d_ibeta * J * sigma_inv * t / Pn;
                      
          }
        else {
                  
            double z = 1 - ibeta(0.5, 0.5 * nu_dbl, r);
            zJacobian *= -1;
                  
            const double Pn = t > 0 ? 0.5 * z : 1.0 - 0.5 * z;
                  
            double d_ibeta = ibeta_derivative(0.5, 0.5 * nu_dbl, r);
                  
            P += log(Pn);
                  
            if (!is_constant_struct<T_y>::value)
              operands_and_partials.d_x1[n] 
                -= zJacobian * d_ibeta * J * sigma_inv / Pn;
                  
            if (!is_constant_struct<T_dof>::value) {
                      
              double g1 = 0;
              double g2 = 0;
                      
              stan::math::gradRegIncBeta(g1, g2, 0.5, 0.5 * nu_dbl, r, 
                                         digammaHalf, digammaNu_vec[n], 
                                         digammaNuPlusHalf_vec[n], 
                                         betaNuHalf_vec[n]);
                      
              operands_and_partials.d_x2[n] 
                -= zJacobian * ( - d_ibeta * (r / t) * (r / t) + 0.5 * g2 ) / Pn;
            }
                  
            if (!is_constant_struct<T_loc>::value)
              operands_and_partials.d_x3[n] 
                += zJacobian * d_ibeta * J * sigma_inv / Pn;
            if (!is_constant_struct<T_scale>::value)
              operands_and_partials.d_x4[n] 
                += zJacobian * d_ibeta * J * sigma_inv * t / Pn;
          }
      }

      return operands_and_partials.to_var(P);
    }

    template <class RNG>
    inline double
    student_t_rng(const double nu,
                  const double mu,
                  const double sigma,
                  RNG& rng) {
      using boost::variate_generator;
      using boost::random::student_t_distribution;

      static const std::string function("stan::prob::student_t_rng");

      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_finite;

      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      variate_generator<RNG&, student_t_distribution<> >
        rng_unit_student_t(rng, student_t_distribution<>(nu));
      return mu + sigma * rng_unit_student_t();
    }
  }
}
#endif
