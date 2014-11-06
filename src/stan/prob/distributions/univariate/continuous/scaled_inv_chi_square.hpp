#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__SCALED_INV_CHI_SQUARE_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__SCALED_INV_CHI_SQUARE_HPP

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/square.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/internal_math.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of a scaled inverse chi-squared density for y with the
     * specified degrees of freedom parameter and scale parameter.
     * 
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Inv-}}\chi^2(\nu, s^2) \\
     \log (p (y \,|\, \nu, s)) &=& \log \left( \frac{(\nu / 2)^{\nu / 2}}{\Gamma (\nu / 2)} s^\nu y^{- (\nu / 2 + 1)} \exp^{-\nu s^2 / (2y)} \right) \\
     &=& \frac{\nu}{2} \log(\frac{\nu}{2}) - \log (\Gamma (\nu / 2)) + \nu \log(s) - (\frac{\nu}{2} + 1) \log(y) - \frac{\nu s^2}{2y} \\
     & & \mathrm{ where } \; y > 0
     \f}
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param s Scale parameter.
     * @throw std::domain_error if nu is not greater than 0
     * @throw std::domain_error if s is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <bool propto,
              typename T_y, typename T_dof, typename T_scale>
    typename return_type<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s) {
      static const std::string function("stan::prob::scaled_inv_chi_square_log");
      
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(nu) 
            && stan::length(s)))
        return 0.0;

      double logp(0.0);
      check_not_nan(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_positive_finite(function, "Scale parameter", s);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu,
                             "Scale parameter", s);

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_dof,T_scale>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_scale> s_vec(s);
      size_t N = max_size(y, nu, s);

      for (size_t n = 0; n < N; n++) {
        if (value_of(y_vec[n]) <= 0)
          return LOG_ZERO;
      }

      using boost::math::lgamma;
      using boost::math::digamma;
      using stan::math::multiply_log;
      using stan::math::square;
      
      DoubleVectorView<include_summand<propto,T_dof,T_y,T_scale>::value,
        is_vector<T_dof>::value> half_nu(length(nu));
      for (size_t i = 0; i < length(nu); i++)
        if (include_summand<propto,T_dof,T_y,T_scale>::value)
          half_nu[i] = 0.5 * value_of(nu_vec[i]);

      DoubleVectorView<include_summand<propto,T_dof,T_y>::value,
        is_vector<T_y>::value> log_y(length(y));      
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_dof,T_y>::value)
          log_y[i] = log(value_of(y_vec[i]));

      DoubleVectorView<include_summand<propto,T_dof,T_y,T_scale>::value,
        is_vector<T_y>::value> inv_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_dof,T_y,T_scale>::value)
          inv_y[i] = 1.0 / value_of(y_vec[i]);
      
      DoubleVectorView<include_summand<propto,T_dof,T_scale>::value,
        is_vector<T_scale>::value> log_s(length(s));
      for (size_t i = 0; i < length(s); i++)
        if (include_summand<propto,T_dof,T_scale>::value)
          log_s[i] = log(value_of(s_vec[i]));
      
      DoubleVectorView<include_summand<propto,T_dof>::value,
        is_vector<T_dof>::value> log_half_nu(length(nu));
      DoubleVectorView<include_summand<propto,T_dof>::value,
        is_vector<T_dof>::value> lgamma_half_nu(length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
        is_vector<T_dof>::value> digamma_half_nu_over_two(length(nu));
      for (size_t i = 0; i < length(nu); i++) {
        if (include_summand<propto,T_dof>::value)
          lgamma_half_nu[i] = lgamma(half_nu[i]);
        if (include_summand<propto,T_dof>::value)
          log_half_nu[i] = log(half_nu[i]);
        if (!is_constant_struct<T_dof>::value)
          digamma_half_nu_over_two[i] = digamma(half_nu[i]) * 0.5;
      }

      agrad::OperandsAndPartials<T_y,T_dof,T_scale> 
        operands_and_partials(y, nu, s);
      for (size_t n = 0; n < N; n++) {
        const double s_dbl = value_of(s_vec[n]);
        const double nu_dbl = value_of(nu_vec[n]);
        if (include_summand<propto,T_dof>::value) 
          logp += half_nu[n] * log_half_nu[n] - lgamma_half_nu[n];
        if (include_summand<propto,T_dof,T_scale>::value)
          logp += nu_dbl * log_s[n];
        if (include_summand<propto,T_dof,T_y>::value)
          logp -= (half_nu[n]+1.0) * log_y[n];
        if (include_summand<propto,T_dof,T_y,T_scale>::value)
          logp -= half_nu[n] * s_dbl*s_dbl * inv_y[n];

        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n] 
            += -(half_nu[n] + 1.0) * inv_y[n] 
            + half_nu[n] * s_dbl*s_dbl * inv_y[n]*inv_y[n];
        }
        if (!is_constant_struct<T_dof>::value) {
          operands_and_partials.d_x2[n] 
            += 0.5 * log_half_nu[n] + 0.5
            - digamma_half_nu_over_two[n]
            + log_s[n]
            - 0.5 * log_y[n]
            - 0.5* s_dbl*s_dbl * inv_y[n];
        }
        if (!is_constant_struct<T_scale>::value) {
          operands_and_partials.d_x3[n] 
            += nu_dbl / s_dbl - nu_dbl * inv_y[n] * s_dbl;
        }
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_dof, typename T_scale>
    inline
    typename return_type<T_y,T_dof,T_scale>::type
    scaled_inv_chi_square_log(const T_y& y, const T_dof& nu, const T_scale& s) {
      return scaled_inv_chi_square_log<false>(y,nu,s);
    }
      
    /**
     * The CDF of a scaled inverse chi-squared density for y with the
     * specified degrees of freedom parameter and scale parameter.
     * 
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @param s Scale parameter.
     * @throw std::domain_error if nu is not greater than 0
     * @throw std::domain_error if s is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
      
    template <typename T_y, typename T_dof, typename T_scale>
    typename return_type<T_y, T_dof, T_scale>::type
    scaled_inv_chi_square_cdf(const T_y& y, const T_dof& nu, 
                              const T_scale& s) {
      // Size checks
      if (!(stan::length(y) && stan::length(nu) && stan::length(s)))
        return 1.0;
      
      static const std::string function("stan::prob::scaled_inv_chi_square_cdf");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
          
      double P(1.0);
          
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_positive_finite(function, "Scale parameter", s);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Degrees of freedom parameter", nu, 
                             "Scale parameter", s);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_scale> s_vec(s);
      size_t N = max_size(y, nu, s);
          
      agrad::OperandsAndPartials<T_y, T_dof, T_scale> 
        operands_and_partials(y, nu, s);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
          
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(0.0);
      }
          
      // Compute CDF and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_q;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> gamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double half_nu_dbl = 0.5 * value_of(nu_vec[i]);
          gamma_vec[i] = tgamma(half_nu_dbl);
          digamma_vec[i] = digamma(half_nu_dbl);
        }
      }
          
      // Compute vectorized CDF and gradient
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double y_inv_dbl = 1.0 / y_dbl;
        const double half_nu_dbl = 0.5 * value_of(nu_vec[n]);
        const double s_dbl = value_of(s_vec[n]);
        const double half_s2_overx_dbl = 0.5 * s_dbl * s_dbl * y_inv_dbl;
        const double half_nu_s2_overx_dbl 
          = 2.0 * half_nu_dbl * half_s2_overx_dbl;
                    
        // Compute
        const double Pn = gamma_q(half_nu_dbl, half_nu_s2_overx_dbl);
                    
        P *= Pn;
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += half_nu_s2_overx_dbl * y_inv_dbl 
            * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) / Pn;
                    
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            += (0.5 * stan::math::gradRegIncGamma(half_nu_dbl,
                                                  half_nu_s2_overx_dbl,
                                                  gamma_vec[n], digamma_vec[n])
                - half_s2_overx_dbl 
                * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) )
            / Pn;
                    
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += - 2.0 * half_nu_dbl * s_dbl * y_inv_dbl 
            * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) / Pn;
              
      }
          
      if (!is_constant_struct<T_y>::value)
        for(size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= P;
      if (!is_constant_struct<T_dof>::value)
        for(size_t n = 0; n < stan::length(nu); ++n) 
          operands_and_partials.d_x2[n] *= P;
      if (!is_constant_struct<T_scale>::value)
        for(size_t n = 0; n < stan::length(s); ++n) 
          operands_and_partials.d_x3[n] *= P;
          
      return operands_and_partials.to_var(P);
    }
      
    template <typename T_y, typename T_dof, typename T_scale>
    typename return_type<T_y, T_dof, T_scale>::type
    scaled_inv_chi_square_cdf_log(const T_y& y, const T_dof& nu, 
                                  const T_scale& s) {
      // Size checks
      if (!(stan::length(y) && stan::length(nu) && stan::length(s)))
        return 0.0;
      
      static const std::string function("stan::prob::scaled_inv_chi_square_cdf_log");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
          
      double P(0.0);
          
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_positive_finite(function, "Scale parameter", s);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Degrees of freedom parameter", nu, 
                             "Scale parameter", s);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_scale> s_vec(s);
      size_t N = max_size(y, nu, s);
          
      agrad::OperandsAndPartials<T_y, T_dof, T_scale> 
        operands_and_partials(y, nu, s);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(stan::math::negative_infinity());
      }
          
      // Compute cdf_log and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_q;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> gamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double half_nu_dbl = 0.5 * value_of(nu_vec[i]);
          gamma_vec[i] = tgamma(half_nu_dbl);
          digamma_vec[i] = digamma(half_nu_dbl);
        }
      }
          
      // Compute vectorized cdf_log and gradient
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          continue;
        }
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double y_inv_dbl = 1.0 / y_dbl;
        const double half_nu_dbl = 0.5 * value_of(nu_vec[n]);
        const double s_dbl = value_of(s_vec[n]);
        const double half_s2_overx_dbl = 0.5 * s_dbl * s_dbl * y_inv_dbl;
        const double half_nu_s2_overx_dbl 
          = 2.0 * half_nu_dbl * half_s2_overx_dbl;
                    
        // Compute
        const double Pn = gamma_q(half_nu_dbl, half_nu_s2_overx_dbl);
                    
        P += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += half_nu_s2_overx_dbl * y_inv_dbl 
            * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            += (0.5 * stan::math::gradRegIncGamma(half_nu_dbl,
                                                  half_nu_s2_overx_dbl,
                                                  gamma_vec[n], digamma_vec[n])
                - half_s2_overx_dbl 
                * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) )
            / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += - 2.0 * half_nu_dbl * s_dbl * y_inv_dbl 
            * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) / Pn;
      }

      return operands_and_partials.to_var(P);
    }      

    template <typename T_y, typename T_dof, typename T_scale>
    typename return_type<T_y, T_dof, T_scale>::type
    scaled_inv_chi_square_ccdf_log(const T_y& y, const T_dof& nu, 
                                   const T_scale& s) {
      // Size checks
      if (!(stan::length(y) && stan::length(nu) && stan::length(s)))
        return 0.0;
      
      static const std::string function("stan::prob::scaled_inv_chi_square_ccdf_log");
          
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_nonnegative;
      using stan::math::value_of;
          
      double P(0.0);
          
      check_not_nan(function, "Random variable", y);
      check_nonnegative(function, "Random variable", y);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_positive_finite(function, "Scale parameter", s);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Degrees of freedom parameter", nu, 
                             "Scale parameter", s);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      VectorView<const T_scale> s_vec(s);
      size_t N = max_size(y, nu, s);
          
      agrad::OperandsAndPartials<T_y, T_dof, T_scale> 
        operands_and_partials(y, nu, s);
          
      // Explicit return for extreme values
      // The gradients are technically ill-defined, but treated as zero
      for (size_t i = 0; i < stan::length(y); i++) {
        if (value_of(y_vec[i]) == 0) 
          return operands_and_partials.to_var(0.0);
      }
          
      // Compute cdf_log and its gradients
      using boost::math::gamma_p_derivative;
      using boost::math::gamma_q;
      using boost::math::digamma;
      using boost::math::tgamma;
          
      // Cache a few expensive function calls if nu is a parameter
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> gamma_vec(stan::length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
                       is_vector<T_dof>::value> digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value) {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const double half_nu_dbl = 0.5 * value_of(nu_vec[i]);
          gamma_vec[i] = tgamma(half_nu_dbl);
          digamma_vec[i] = digamma(half_nu_dbl);
        }
      }
          
      // Compute vectorized cdf_log and gradient
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          return operands_and_partials.to_var(stan::math::negative_infinity());
        }
              
        // Pull out values
        const double y_dbl = value_of(y_vec[n]);
        const double y_inv_dbl = 1.0 / y_dbl;
        const double half_nu_dbl = 0.5 * value_of(nu_vec[n]);
        const double s_dbl = value_of(s_vec[n]);
        const double half_s2_overx_dbl = 0.5 * s_dbl * s_dbl * y_inv_dbl;
        const double half_nu_s2_overx_dbl 
          = 2.0 * half_nu_dbl * half_s2_overx_dbl;
                    
        // Compute
        const double Pn = 1.0 - gamma_q(half_nu_dbl, half_nu_s2_overx_dbl);
                    
        P += log(Pn);
              
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            -= half_nu_s2_overx_dbl * y_inv_dbl 
            * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            -= (0.5 * stan::math::gradRegIncGamma(half_nu_dbl,
                                                  half_nu_s2_overx_dbl,
                                                  gamma_vec[n], digamma_vec[n])
                - half_s2_overx_dbl 
                * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) )
            / Pn;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += 2.0 * half_nu_dbl * s_dbl * y_inv_dbl 
            * gamma_p_derivative(half_nu_dbl, half_nu_s2_overx_dbl) / Pn;
      }

      return operands_and_partials.to_var(P);
    }  

    template <class RNG>
    inline double
    scaled_inv_chi_square_rng(const double nu,
                              const double s,
                              RNG& rng) {
      using boost::variate_generator;
      using boost::random::chi_squared_distribution;

      static const std::string function("stan::prob::scaled_inv_chi_square_rng");
      
      using stan::error_handling::check_positive_finite;

      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_positive_finite(function, "Scale parameter", s);

      variate_generator<RNG&, chi_squared_distribution<> >
        chi_square_rng(rng, chi_squared_distribution<>(nu));
      return nu * s / chi_square_rng();
    }    
  }
}
#endif

