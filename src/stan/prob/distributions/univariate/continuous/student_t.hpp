#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__STUDENT_T_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__STUDENT_T_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/internal_math.hpp>

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
              typename T_loc, typename T_scale,
              class Policy>
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma, 
                  const Policy&) {
      static const char* function = "stan::prob::student_t_log(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(nu) 
            && stan::length(mu)
	    && stan::length(sigma)))
        return 0.0;

      typename return_type<T_y,T_dof,T_loc,T_scale>::type logp = 0.0;

      // validate args (here done over var, which should be OK)
      if (!check_not_nan(function, y, "Random variable", &logp, Policy()))
        return logp;
      if(!check_finite(function, nu, "Degrees of freedom parameter", &logp, Policy()))
        return logp;
      if(!check_positive(function, nu, "Degrees of freedom parameter", &logp, Policy()))
        return logp;
      if (!check_finite(function, mu, "Location parameter", 
                        &logp, Policy()))
        return logp;
      if (!check_finite(function, sigma, "Scale parameter", 
                        &logp, Policy()))
        return logp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &logp, Policy()))
        return logp;

      
      if (!(check_consistent_sizes(function,
                                   y,nu,mu,sigma,
				   "Random variable","Degrees of freedom parameter","Location parameter","Scale parameter",
                                   &logp, Policy())))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
	return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, nu, mu, sigma);

      using stan::math::square;
      using boost::math::lgamma;

      using std::log;

      for (size_t n = 0; n < N; n++) {
	if (include_summand<propto,T_dof>::value)
	  logp += lgamma( (nu_vec[n] + 1.0) / 2.0) - lgamma(nu_vec[n] / 2.0);
	if (include_summand<propto>::value)
	  logp += NEG_LOG_SQRT_PI;
	if (include_summand<propto,T_dof>::value)
	  logp -= 0.5 * log(nu_vec[n]);
	if (include_summand<propto,T_scale>::value)
	  logp -= log(sigma_vec[n]);
	if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value)
	  logp -= ((nu_vec[n] + 1.0) / 2.0) 
	    * log1p( square(((y_vec[n] - mu_vec[n]) / sigma_vec[n])) / nu_vec[n]);
      }
      return logp;
    }

    template <bool propto, 
              typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
      return student_t_log<propto>(y,nu,mu,sigma,stan::math::default_policy());
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale,
              class Policy>
    inline
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma, 
                  const Policy&) {
      return student_t_log<false>(y,nu,mu,sigma,Policy());
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y,T_dof,T_loc,T_scale>::type
    student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu, 
                  const T_scale& sigma) {
      return student_t_log<false>(y,nu,mu,sigma,stan::math::default_policy());
    }
      
      template <typename T_y, typename T_dof, typename T_loc, typename T_scale, class Policy>
      typename boost::math::tools::promote_args<T_y, T_dof, T_loc, T_scale>::type
      student_t_cdf(const T_y& y, const T_dof& nu, const T_loc& mu, const T_scale& sigma, const Policy&) {
          
          static const char* function = "stan::prob::student_t_cdf(%1%)";
          
          using stan::math::check_positive;
          using stan::math::check_finite;
          using stan::math::check_not_nan;
          using stan::math::check_consistent_sizes;
          
          using boost::math::tools::promote_args;
          
          
          double P(1.0);
          
          if (!check_not_nan(function, y, "Random variable", &P, Policy()))
              return P;
          
          if(!check_finite(function, nu, "Degrees of freedom parameter", &P, Policy()))
              return P;
          
          if(!check_positive(function, nu, "Degrees of freedom parameter", &P, Policy()))
              return P;
          
          if (!check_finite(function, mu, "Location parameter", 
                            &P, Policy()))
              return P;
          
          if (!check_finite(function, sigma, "Scale parameter", 
                            &P, Policy()))
              return P;
          if (!check_positive(function, sigma, "Scale parameter", 
                              &P, Policy()))
              return P;
          
          // Wrap arguments in vectors
          VectorView<const T_y> y_vec(y);
          VectorView<const T_dof> nu_vec(nu);
          VectorView<const T_loc> mu_vec(mu);
          VectorView<const T_scale> sigma_vec(sigma);
          //size_t N = max_size(y, nu, mu, sigma);
          size_t N = max_size(y, nu);
          
          agrad::OperandsAndPartials<T_y, T_dof, T_loc, T_scale> operands_and_partials(y, nu, mu, sigma);
          
          std::fill(operands_and_partials.all_partials,
                    operands_and_partials.all_partials + operands_and_partials.nvaris, 0.0);
          
          using stan::math::value_of;
          using boost::math::ibeta;
          using boost::math::ibeta_derivative;
          
          using boost::math::digamma;
          using boost::math::beta;
          
          // Cache a few expensive function calls if nu is a parameter
          double digammaHalf = 0;
          
          DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> digamma_vec(stan::length(nu));
          DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> digammaNu_vec(stan::length(nu));
          DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> digammaNuPlusHalf_vec(stan::length(nu));
          DoubleVectorView<!is_constant_struct<T_dof>::value,is_vector<T_dof>::value> betaNuHalf_vec(stan::length(nu));
          
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
              
              const double sigma_inv = 1.0 / value_of(sigma_vec[n]);
              const double t = (value_of(y_vec[n]) - value_of(mu_vec[n])) * sigma_inv;
              const double nu_dbl = value_of(nu_vec[n]);
              const double q = nu_dbl / (t * t);
              const double r = 1.0 / (1.0 + q);
              double J = - 2 * r * r * q / t;
              
              if(q < 2)
              {
                  
                  double z = ibeta(0.5 * nu_dbl, 0.5, 1.0 - r);
                  
                  const double Pn = t > 0 ? 1.0 - 0.5 * z : 0.5 * z;
                  
                  double d_ibeta = ibeta_derivative(0.5 * nu_dbl, 0.5, 1.0 - r);
                  
                  P *= Pn;
                  
                  if (!is_constant_struct<T_y>::value)
                      operands_and_partials.d_x1[n] 
                      += (d_ibeta * J * sigma_inv) / Pn;
                  
                  if (!is_constant_struct<T_dof>::value) {
                      
                      double g1 = 0;
                      double g2 = 0;
                      
                      stan::math::gradRegIncBeta(g1, g2, 0.5 * nu_dbl, 0.5, 1.0 - r, 
                                     digammaNu_vec[n], digammaHalf, digammaNuPlusHalf_vec[n], betaNuHalf_vec[n]);
                      
                      operands_and_partials.d_x2[n] 
                      += 0.5 * ( d_ibeta * (r / t) * (r / t) + 0.5 * g1 ) / Pn;
                      
                  }
                  
                  if (!is_constant_struct<T_loc>::value)
                      operands_and_partials.d_x3[n] 
                      += (- d_ibeta * J * sigma_inv) / Pn;
                  
                  if (!is_constant_struct<T_scale>::value)
                      operands_and_partials.d_x4[n] 
                      += (- d_ibeta * J * sigma_inv * t) / Pn;
                  
              }
              else
              {
                  
                  double z = 1.0 - ibeta(0.5, 0.5 * nu_dbl, r);
                  
                  const double Pn = t > 0 ? 1.0 - 0.5 * z : 0.5 * z;
                  
                  double d_ibeta = ibeta_derivative(0.5, 0.5 * nu_dbl, r);
                  
                  P *= Pn;
                  
                  if (!is_constant_struct<T_y>::value)
                      operands_and_partials.d_x1[n] 
                      += (d_ibeta * J * sigma_inv) / Pn;
                  
                  if (!is_constant_struct<T_dof>::value) {
                      
                      double g1 = 0;
                      double g2 = 0;
                      
                      stan::math::gradRegIncBeta(g1, g2, 0.5, 0.5 * nu_dbl, r, 
                                     digammaHalf, digammaNu_vec[n], digammaNuPlusHalf_vec[n], betaNuHalf_vec[n]);
                      
                      operands_and_partials.d_x2[n] 
                      += 0.5 * ( - d_ibeta * (r / t) * (r / t) + 0.5 * g2 ) / Pn;
                      
                  }
                  
                  if (!is_constant_struct<T_loc>::value)
                      operands_and_partials.d_x3[n] 
                      += (- d_ibeta * J * sigma_inv) / Pn;
                  
                  if (!is_constant_struct<T_scale>::value)
                      operands_and_partials.d_x4[n] 
                      += (- d_ibeta * J * sigma_inv * t) / Pn;
                  
              }
              
              if(t > 0) operands_and_partials.d_x2[n] *= -1;
              
          }
          
          for (size_t n = 0; n < N; n++) {
              
              if (!is_constant_struct<T_y>::value)
                  operands_and_partials.d_x1[n] *= P;
              
              if (!is_constant_struct<T_dof>::value)
                  operands_and_partials.d_x2[n] *= P;
              
              if (!is_constant_struct<T_loc>::value)
                  operands_and_partials.d_x3[n] *= P;
              
              if (!is_constant_struct<T_scale>::value)
                  operands_and_partials.d_x4[n] *= P;
              
          }
          
          return operands_and_partials.to_var(P);
          
      }
      
      template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
      typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
      student_t_cdf(const T_y& y, const T_dof& nu, const T_loc& mu, const T_scale& sigma) {
          return student_t_cdf(y, nu,  mu, sigma, stan::math::default_policy());
      }
    
  }
}
#endif
