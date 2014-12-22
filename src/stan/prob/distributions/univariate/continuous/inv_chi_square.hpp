#ifndef STAN__PROB__DIST__UNI__CONTINUOUS__INV_CHI_SQUARE_HPP
#define STAN__PROB__DIST__UNI__CONTINUOUS__INV_CHI_SQUARE_HPP

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/functions/gamma_q.hpp>
#include <stan/math/functions/digamma.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/internal_math/math/grad_reg_inc_gamma.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of an inverse chi-squared density for y with the specified
     * degrees of freedom parameter.
     * The degrees of freedom prarameter must be greater than 0.
     * y must be greater than 0.
     * 
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Inv-}}\chi^2_\nu \\
     \log (p (y \,|\, \nu)) &=& \log \left( \frac{2^{-\nu / 2}}{\Gamma (\nu / 2)} y^{- (\nu / 2 + 1)} \exp^{-1 / (2y)} \right) \\
     &=& - \frac{\nu}{2} \log(2) - \log (\Gamma (\nu / 2)) - (\frac{\nu}{2} + 1) \log(y) - \frac{1}{2y} \\
     & & \mathrm{ where } \; y > 0
     \f}
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <bool propto,
              typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    inv_chi_square_log(const T_y& y, const T_dof& nu) {
      static const std::string function("stan::prob::inv_chi_square_log");
      typedef typename stan::partials_return_type<T_y,T_dof>::type 
        T_partials_return;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(nu)))
        return 0.0;

      using stan::error_handling::check_positive_finite;      
      using stan::error_handling::check_positive;      
      using stan::error_handling::check_not_nan;
      using stan::math::value_of;
      using stan::error_handling::check_consistent_sizes;

      T_partials_return logp(0.0);
      check_positive_finite(function, "Degrees of freedom parameter", nu);
			check_positive(function,"Random variable",y);
      check_not_nan(function, "Random variable", y);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Degrees of freedom parameter", nu);

       
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
      
      using boost::math::digamma;
      using boost::math::lgamma;
      using stan::math::multiply_log;

      VectorBuilder<include_summand<propto,T_y,T_dof>::value,
                    T_partials_return, T_y> log_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y,T_dof>::value)
          log_y[i] = log(value_of(y_vec[i]));

      VectorBuilder<include_summand<propto,T_y>::value,
                    T_partials_return, T_y> inv_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y>::value)
          inv_y[i] = 1.0 / value_of(y_vec[i]);

      VectorBuilder<include_summand<propto,T_dof>::value,
                    T_partials_return, T_dof> lgamma_half_nu(length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                     T_partials_return, T_dof> 
        digamma_half_nu_over_two(length(nu));
      for (size_t i = 0; i < length(nu); i++) {
        T_partials_return half_nu = 0.5 * value_of(nu_vec[i]);
        if (include_summand<propto,T_dof>::value)
          lgamma_half_nu[i] = lgamma(half_nu);
        if (!is_constant_struct<T_dof>::value)
          digamma_half_nu_over_two[i] = digamma(half_nu) * 0.5;
      }

      agrad::OperandsAndPartials<T_y, T_dof> operands_and_partials(y, nu);
      for (size_t n = 0; n < N; n++) {
        const T_partials_return nu_dbl = value_of(nu_vec[n]);
        const T_partials_return half_nu = 0.5 * nu_dbl;
  
        if (include_summand<propto,T_dof>::value)
          logp += nu_dbl * NEG_LOG_TWO_OVER_TWO - lgamma_half_nu[n];
        if (include_summand<propto,T_y,T_dof>::value)
          logp -= (half_nu+1.0) * log_y[n];
        if (include_summand<propto,T_y>::value)
          logp -= 0.5 * inv_y[n];

        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n] 
            += -(half_nu+1.0) * inv_y[n] + 0.5 * inv_y[n] * inv_y[n];
        }
        if (!is_constant_struct<T_dof>::value) {
          operands_and_partials.d_x2[n]
            += NEG_LOG_TWO_OVER_TWO - digamma_half_nu_over_two[n]
            - 0.5*log_y[n];
        }
      }
      return operands_and_partials.to_var(logp,y,nu);
    }

    template <typename T_y, typename T_dof>
    inline
    typename return_type<T_y,T_dof>::type
    inv_chi_square_log(const T_y& y, const T_dof& nu) {
      return inv_chi_square_log<false>(y,nu);
    }
      
    template <typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    inv_chi_square_cdf(const T_y& y, const T_dof& nu) {
      typedef typename stan::partials_return_type<T_y,T_dof>::type 
        T_partials_return;

      // Size checks
      if ( !( stan::length(y) && stan::length(nu) ) ) return 1.0;
          
      // Error checks
      static const std::string function("stan::prob::inv_chi_square_cdf");
          
      using stan::error_handling::check_positive_finite;      
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_positive;
      using boost::math::tools::promote_args;
      using stan::math::value_of;
          
      T_partials_return P(1.0);
          
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_not_nan(function, "Random variable", y);
      check_positive(function, "Random variable", y);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Degrees of freedom parameter", nu);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
          
      agrad::OperandsAndPartials<T_y, T_dof> operands_and_partials(y, nu);
          
      // Compute CDF and its gradients
      using stan::math::gamma_q;
      using stan::math::digamma;
      using boost::math::tgamma;
      using std::exp;
      using std::pow;
          
      // Cache a few expensive function calls if nu is a parameter
      VectorBuilder<!is_constant_struct<T_dof>::value,
                     T_partials_return, T_dof> gamma_vec(stan::length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value)  {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const T_partials_return nu_dbl = value_of(nu_vec[i]);
          gamma_vec[i] = tgamma(0.5 * nu_dbl);
          digamma_vec[i] = digamma(0.5 * nu_dbl);
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
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return y_inv_dbl = 1.0 / y_dbl;
        const T_partials_return nu_dbl = value_of(nu_vec[n]);
                  
        // Compute
        const T_partials_return Pn = gamma_q(0.5 * nu_dbl, 0.5 * y_inv_dbl);
                  
        P *= Pn;
                  
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += 0.5 * y_inv_dbl * y_inv_dbl
            * exp(-0.5*y_inv_dbl) * pow(0.5*y_inv_dbl,0.5*nu_dbl-1) 
            / tgamma(0.5*nu_dbl) / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            += 0.5 * stan::math::grad_reg_inc_gamma(0.5 * nu_dbl, 
                                                    0.5 * y_inv_dbl, 
                                                    gamma_vec[n], 
                                                    digamma_vec[n]) / Pn;
      }
              
      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= P;
      if (!is_constant_struct<T_dof>::value)
        for (size_t n = 0; n < stan::length(nu); ++n) 
          operands_and_partials.d_x2[n] *= P;
          
      return operands_and_partials.to_var(P,y,nu);
    }

    template <typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    inv_chi_square_cdf_log(const T_y& y, const T_dof& nu) {
      typedef typename stan::partials_return_type<T_y,T_dof>::type
        T_partials_return;

      // Size checks
      if ( !( stan::length(y) && stan::length(nu) ) ) return 0.0;
          
      // Error checks
      static const std::string function("stan::prob::inv_chi_square_cdf_log");
          
      using stan::error_handling::check_positive_finite;      
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_positive;
      using boost::math::tools::promote_args;
      using stan::math::value_of;
          
      T_partials_return P(0.0);
          
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_not_nan(function, "Random variable", y);
      check_positive(function, "Random variable", y);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Degrees of freedom parameter", nu);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
          
      agrad::OperandsAndPartials<T_y, T_dof> operands_and_partials(y, nu);
          
      // Compute cdf_log and its gradients
      using stan::math::gamma_q;
      using stan::math::digamma;
      using boost::math::tgamma;
      using std::exp;
      using std::pow;
          
      // Cache a few expensive function calls if nu is a parameter
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> gamma_vec(stan::length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value)  {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const T_partials_return nu_dbl = value_of(nu_vec[i]);
          gamma_vec[i] = tgamma(0.5 * nu_dbl);
          digamma_vec[i] = digamma(0.5 * nu_dbl);
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
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return y_inv_dbl = 1.0 / y_dbl;
        const T_partials_return nu_dbl = value_of(nu_vec[n]);
                  
        // Compute
        const T_partials_return Pn = gamma_q(0.5 * nu_dbl, 0.5 * y_inv_dbl);
                  
        P += log(Pn);
                  
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += 0.5 * y_inv_dbl * y_inv_dbl
            * exp(-0.5*y_inv_dbl) * pow(0.5*y_inv_dbl,0.5*nu_dbl-1) 
            / tgamma(0.5*nu_dbl) / Pn;
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            += 0.5 * stan::math::grad_reg_inc_gamma(0.5 * nu_dbl, 
                                                    0.5 * y_inv_dbl, 
                                                    gamma_vec[n], 
                                                    digamma_vec[n]) / Pn;
      }
              
      return operands_and_partials.to_var(P,y,nu);
    }
      
    template <typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    inv_chi_square_ccdf_log(const T_y& y, const T_dof& nu) {
      typedef typename stan::partials_return_type<T_y,T_dof>::type
        T_partials_return;

      // Size checks
      if ( !( stan::length(y) && stan::length(nu) ) ) return 0.0;
          
      // Error checks
      static const std::string function("stan::prob::inv_chi_square_ccdf_log");
          
      using stan::error_handling::check_positive_finite;      
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_consistent_sizes;
      using stan::error_handling::check_positive;
      using boost::math::tools::promote_args;
      using stan::math::value_of;
          
      T_partials_return P(0.0);
          
      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_not_nan(function, "Random variable", y);
      check_positive(function, "Random variable", y);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Degrees of freedom parameter", nu);
          
      // Wrap arguments in vectors
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
          
      agrad::OperandsAndPartials<T_y, T_dof> operands_and_partials(y, nu);
			
      // Compute ccdf_log and its gradients
      using stan::math::gamma_q;
      using stan::math::digamma;
      using boost::math::tgamma;
      using std::exp;
      using std::pow;

      // Cache a few expensive function calls if nu is a parameter
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> gamma_vec(stan::length(nu));
      VectorBuilder<!is_constant_struct<T_dof>::value,
                    T_partials_return, T_dof> digamma_vec(stan::length(nu));
          
      if (!is_constant_struct<T_dof>::value)  {
        for (size_t i = 0; i < stan::length(nu); i++) {
          const T_partials_return nu_dbl = value_of(nu_vec[i]);
          gamma_vec[i] = tgamma(0.5 * nu_dbl);
          digamma_vec[i] = digamma(0.5 * nu_dbl);
        }
      }
          
      // Compute vectorized ccdf_log and gradient
      for (size_t n = 0; n < N; n++) {
              
        // Explicit results for extreme values
        // The gradients are technically ill-defined, but treated as zero
        if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
          return operands_and_partials.to_var(stan::math::negative_infinity(),
                                              y,nu);
        }

        // Pull out values
        const T_partials_return y_dbl = value_of(y_vec[n]);
        const T_partials_return y_inv_dbl = 1.0 / y_dbl;
        const T_partials_return nu_dbl = value_of(nu_vec[n]);
                  
        // Compute
        const T_partials_return Pn = 1.0 - gamma_q(0.5 * nu_dbl, 0.5 
                                                   * y_inv_dbl);
                  
        P += log(Pn);
                  
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= 0.5 * y_inv_dbl * y_inv_dbl
            * exp(-0.5*y_inv_dbl) * pow(0.5*y_inv_dbl,0.5*nu_dbl-1) 
            / tgamma(0.5*nu_dbl) / Pn;      
        if (!is_constant_struct<T_dof>::value)
          operands_and_partials.d_x2[n] 
            -= 0.5 * stan::math::grad_reg_inc_gamma(0.5 * nu_dbl, 
                                                    0.5 * y_inv_dbl, 
                                                    gamma_vec[n], 
                                                    digamma_vec[n]) / Pn;
      }
              
      return operands_and_partials.to_var(P,y,nu);
    }

    template <class RNG>
    inline double
    inv_chi_square_rng(const double nu,
                       RNG& rng) {
      using boost::variate_generator;
      using boost::random::chi_squared_distribution;

      static const std::string function("stan::prob::inv_chi_square_rng");

      using stan::error_handling::check_positive_finite;      

      check_positive_finite(function, "Degrees of freedom parameter", nu);

      variate_generator<RNG&, chi_squared_distribution<> >
        chi_square_rng(rng, chi_squared_distribution<>(nu));
      return 1 / chi_square_rng();
    }
    
  }
}

#endif

