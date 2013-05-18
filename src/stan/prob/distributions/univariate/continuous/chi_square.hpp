#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__CHI_SQUARE_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__CHI_SQUARE_HPP__

#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * The log of a chi-squared density for y with the specified
     * degrees of freedom parameter.
     * The degrees of freedom prarameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
     y &\sim& \chi^2_\nu \\
     \log (p (y \,|\, \nu)) &=& \log \left( \frac{2^{-\nu / 2}}{\Gamma (\nu / 2)} y^{\nu / 2 - 1} \exp^{- y / 2} \right) \\
     &=& - \frac{\nu}{2} \log(2) - \log (\Gamma (\nu / 2)) + (\frac{\nu}{2} - 1) \log(y) - \frac{y}{2} \\
     & & \mathrm{ where } \; y \ge 0
     \f}
     * @param y A scalar variable.
     * @param nu Degrees of freedom.
     * @throw std::domain_error if nu is not greater than or equal to 0
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     */
    template <bool propto,
              typename T_y, typename T_dof>
    typename return_type<T_y,T_dof>::type
    chi_square_log(const T_y& y, const T_dof& nu) {
      static const char* function = "stan::prob::chi_square_log(%1%)";

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(nu)))
        return 0.0;
      
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      
      double logp(0.0);
      if (!check_not_nan(function, y, "Random variable", &logp))
        return logp;
      if (!check_finite(function, nu, "Degrees of freedom parameter", &logp))
        return logp;
      if (!check_positive(function, nu, "Degrees of freedom parameter", &logp))
        return logp;
    
      if (!(check_consistent_sizes(function,
                                   y,nu,
                                   "Random variable","Degrees of freedom parameter",
                                   &logp)))
        return logp;
    
      
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_dof> nu_vec(nu);
      size_t N = max_size(y, nu);
      
      for (size_t n = 0; n < length(y); n++) 
        if (value_of(y_vec[n]) < 0)
          return LOG_ZERO;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_dof>::value)
        return 0.0;

      using boost::math::digamma;
      using boost::math::lgamma;
      using stan::math::multiply_log;

      DoubleVectorView<include_summand<propto,T_y,T_dof>::value,
        is_vector<T_y>::value> log_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y,T_dof>::value)
          log_y[i] = log(value_of(y_vec[i]));

      DoubleVectorView<include_summand<propto,T_y>::value,
        is_vector<T_y>::value> inv_y(length(y));
      for (size_t i = 0; i < length(y); i++)
        if (include_summand<propto,T_y>::value)
          inv_y[i] = 1.0 / value_of(y_vec[i]);

      DoubleVectorView<include_summand<propto,T_dof>::value,
        is_vector<T_dof>::value> lgamma_half_nu(length(nu));
      DoubleVectorView<!is_constant_struct<T_dof>::value,
        is_vector<T_dof>::value> digamma_half_nu_over_two(length(nu));

      for (size_t i = 0; i < length(nu); i++) {
        double half_nu = 0.5 * value_of(nu_vec[i]);
        if (include_summand<propto,T_dof>::value)
          lgamma_half_nu[i] = lgamma(half_nu);
        if (!is_constant_struct<T_dof>::value)
          digamma_half_nu_over_two[i] = digamma(half_nu) * 0.5;
      }


      agrad::OperandsAndPartials<T_y,T_dof> operands_and_partials(y, nu);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double half_y = 0.5 * y_dbl;
        const double nu_dbl = value_of(nu_vec[n]);
        const double half_nu = 0.5 * nu_dbl;
        if (include_summand<propto,T_dof>::value)
          logp += nu_dbl * NEG_LOG_TWO_OVER_TWO - lgamma_half_nu[n];
        if (include_summand<propto,T_y,T_dof>::value)
          logp += (half_nu-1.0) * log_y[n];
        if (include_summand<propto,T_y>::value)
          logp -= half_y;
  
        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n] += (half_nu-1.0)*inv_y[n] - 0.5;
        }
        if (!is_constant_struct<T_dof>::value) {
          operands_and_partials.d_x2[n] 
            += NEG_LOG_TWO_OVER_TWO - digamma_half_nu_over_two[n] + log_y[n]*0.5; 
        }
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_dof>
    inline
    typename return_type<T_y,T_dof>::type
    chi_square_log(const T_y& y, const T_dof& nu) {
      return chi_square_log<false>(y,nu);
    }

    /** 
     * Calculates the chi square cumulative distribution function for the given
     * variate and degrees of freedom.
     * 
     * @param y A scalar variate.
     * @param nu Degrees of freedom.
     * 
     * @return The cdf of the chi square distribution
     */
    /*template <typename T_y, typename T_dof>
      typename return_type<T_y,T_dof>::type
      chi_square_cdf(const T_y& y, const T_dof& nu) {
      static const char* function = "stan::prob::chi_square_cdf(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using return_type;

      typename return_type<T_y,T_dof>::type lp;
      if (!check_not_nan(function, y, "Random variable", &lp))
      return lp;
      if (!check_finite(function, nu, "Degrees of freedom parameter", &lp))
      return lp;
      if (!check_positive(function, nu, "Degrees of freedom parameter", &lp))
      return lp;
      
      // FIXME: include when gamma_cdf() is ready
      return stan::prob::gamma_cdf(y,nu/2,0.5);
      }

    */

    template <class RNG>
    inline double
    chi_square_rng(const double nu,
                   RNG& rng) {
      using boost::variate_generator;
      using boost::random::chi_squared_distribution;
      variate_generator<RNG&, chi_squared_distribution<> >
        chi_square_rng(rng, chi_squared_distribution<>(nu));
      return chi_square_rng();
    }
  }
}

#endif

