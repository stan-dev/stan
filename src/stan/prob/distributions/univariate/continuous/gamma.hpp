#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__GAMMA_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__GAMMA_HPP__

#include <boost/random/gamma_distribution.hpp>
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
     * The log of a gamma density for y with the specified
     * shape and inverse scale parameters.
     * Shape and inverse scale parameters must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Gamma}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\beta^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} \exp^{- \beta y} \right) \\
     &=& \alpha \log(\beta) - \log(\Gamma(\alpha)) + (\alpha - 1) \log(y) - \beta y\\
     & & \mathrm{where} \; y > 0
     \f}
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <bool propto,
              typename T_y, typename T_shape, typename T_inv_scale>
    typename return_type<T_y,T_shape,T_inv_scale>::type
    gamma_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      static const char* function = "stan::prob::gamma_log(%1%)";

      using stan::is_constant_struct;
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      if (!check_not_nan(function, y, "Random variable", &logp))
        return logp;
      if (!check_finite(function, alpha, "Shape parameter", 
                        &logp)) 
        return logp;
      if (!check_positive(function, alpha, "Shape parameter", 
                          &logp)) 
        return logp;
      if (!check_finite(function, beta, "Inverse scale parameter", 
                        &logp)) 
        return logp;
      if (!check_positive(function, beta, "Inverse scale parameter", 
                          &logp)) 
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,alpha,beta,
                                   "Random variable","Shape parameter","Inverse scale parameter",
                                   &logp)))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_shape,T_inv_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_shape> alpha_vec(alpha);
      VectorView<const T_inv_scale> beta_vec(beta);

      for (size_t n = 0; n < length(y); n++) {
        const double y_dbl = value_of(y_vec[n]);
        if (y_dbl < 0)
          return LOG_ZERO;
      }

      size_t N = max_size(y, alpha, beta);
      agrad::OperandsAndPartials<T_y, T_shape, T_inv_scale> operands_and_partials(y, alpha, beta);

      using boost::math::lgamma;
      using stan::math::multiply_log;
      using boost::math::digamma;
      
      DoubleVectorView<include_summand<propto,T_y,T_shape>::value,is_vector<T_y>::value>
        log_y(length(y));
      if (include_summand<propto,T_y,T_shape>::value)
        for(size_t n = 0; n < length(y); n++) {
          if (value_of(y_vec[n]) > 0)
            log_y[n] = log(value_of(y_vec[n]));
        }

      DoubleVectorView<include_summand<propto,T_shape>::value,is_vector<T_shape>::value>
        lgamma_alpha(length(alpha));
      DoubleVectorView<!is_constant_struct<T_shape>::value,is_vector<T_shape>::value>
        digamma_alpha(length(alpha));
      for (size_t n = 0; n < length(alpha); n++) {
        if (include_summand<propto,T_shape>::value)
          lgamma_alpha[n] = lgamma(value_of(alpha_vec[n]));
        if (!is_constant_struct<T_shape>::value)
          digamma_alpha[n] = digamma(value_of(alpha_vec[n]));
      }

      DoubleVectorView<include_summand<propto,T_shape,T_inv_scale>::value,is_vector<T_inv_scale>::value>
        log_beta(length(beta));
      if (include_summand<propto,T_shape,T_inv_scale>::value)
        for (size_t n = 0; n < length(beta); n++)
          log_beta[n] = log(value_of(beta_vec[n]));
      
      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
  
        if (include_summand<propto,T_shape>::value)
          logp -= lgamma_alpha[n];
        if (include_summand<propto,T_shape,T_inv_scale>::value)
          logp += alpha_dbl * log_beta[n];
        if (include_summand<propto,T_y,T_shape>::value)
          logp += (alpha_dbl-1.0) * log_y[n];
        if (include_summand<propto,T_y,T_inv_scale>::value)
          logp -= beta_dbl * y_dbl;
  
        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += (alpha_dbl-1)/y_dbl - beta_dbl;
        if (!is_constant_struct<T_shape>::value)
          operands_and_partials.d_x2[n] += -digamma_alpha[n] + log_beta[n] + log_y[n];
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x3[n] += alpha_dbl / beta_dbl - y_dbl;
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_shape, typename T_inv_scale>
    inline
    typename return_type<T_y,T_shape,T_inv_scale>::type
    gamma_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      return gamma_log<false>(y,alpha,beta);
    }


    /**
     * The cumulative density function for a gamma distribution for y
     * with the specified shape and inverse scale parameters.
     *
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */    
    /*template <typename T_y, typename T_shape, typename T_inv_scale>
      typename boost::math::tools::promote_args<T_y,T_shape,T_inv_scale>::type
      gamma_cdf(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      static const char* function = "stan::prob::gamma_cdf(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_shape,T_inv_scale>::type result;
      if (!check_finite(function, y, "Random variable", &result))
      return result;
      if (!check_nonnegative(function, y, "Random variable", &result))
      return result;
      if (!check_finite(function, alpha, "Shape parameter", &result)) 
      return result;
      if (!check_positive(function, alpha, "Shape parameter", &result)) 
      return result;
      if (!check_finite(function, beta, "Inverse scale parameter", 
      &result)) 
      return result;
      if (!check_positive(function, beta, "Inverse scale parameter", 
      &result)) 
      return result;
      
      // FIXME: implement gamma cdf
      return boost::math::gamma_p(alpha, y*beta);
      }

    */

    template <class RNG>
    inline double
    gamma_rng(const double alpha,
              const double beta,
              RNG& rng) {
      using boost::variate_generator;
      using boost::gamma_distribution;
      variate_generator<RNG&, gamma_distribution<> >
        gamma_rng(rng, gamma_distribution<>(alpha, beta));
      return gamma_rng();
    }

  }
}

#endif
