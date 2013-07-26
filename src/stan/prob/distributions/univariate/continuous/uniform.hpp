#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__UNIFORM_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__UNIFORM_HPP__

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // CONTINUOUS, UNIVARIATE DENSITIES
    /**
     * The log of a uniform density for the given 
     * y, lower, and upper bound. 
     *
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{U}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta)) &=& \log \left( \frac{1}{\beta-\alpha} \right) \\
     &=& \log (1) - \log (\beta - \alpha) \\
     &=& -\log (\beta - \alpha) \\
     & & \mathrm{ where } \; y \in [\alpha, \beta], \log(0) \; \mathrm{otherwise}
     \f}
     * 
     * @param y A scalar variable.
     * @param alpha Lower bound.
     * @param beta Upper bound.
     * @throw std::invalid_argument if the lower bound is greater than 
     *    or equal to the lower bound
     * @tparam T_y Type of scalar.
     * @tparam T_low Type of lower bound.
     * @tparam T_high Type of upper bound.
     */
    template <bool propto,
              typename T_y, typename T_low, typename T_high>
    typename return_type<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      static const char* function = "stan::prob::uniform_log(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_greater;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);
      if(!check_not_nan(function, y, "Random variable", &logp))
        return logp;
      if (!check_finite(function, alpha, "Lower bound parameter", &logp))
        return logp;
      if (!check_finite(function, beta, "Upper bound parameter", &logp))
        return logp;
      if (!check_greater(function, beta, alpha, "Upper bound parameter",
                         &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,alpha,beta,
                                   "Random variable","Lower bound parameter",
                                   "Upper bound parameter",
                                   &logp)))
        return logp;

      
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_low,T_high>::value)
        return 0.0;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_low> alpha_vec(alpha);
      VectorView<const T_high> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        if (y_dbl < value_of(alpha_vec[n]) 
            || y_dbl > value_of(beta_vec[n]))
          return LOG_ZERO;
      }

      DoubleVectorView<include_summand<propto,T_low,T_high>::value,
        is_vector<T_low>::value | is_vector<T_high>::value> 
        inv_beta_minus_alpha(max_size(alpha,beta));
      for (size_t i = 0; i < max_size(alpha,beta); i++) 
        if (include_summand<propto,T_low,T_high>::value)
          inv_beta_minus_alpha[i] 
            = 1.0 / (value_of(beta_vec[i]) - value_of(alpha_vec[i]));
      DoubleVectorView<include_summand<propto,T_low,T_high>::value,
        is_vector<T_low>::value | is_vector<T_high>::value> 
        log_beta_minus_alpha(max_size(alpha,beta));
      for (size_t i = 0; i < max_size(alpha,beta); i++)
        if (include_summand<propto,T_low,T_high>::value)
          log_beta_minus_alpha[i] 
            = log(value_of(beta_vec[i]) - value_of(alpha_vec[i]));
      
      agrad::OperandsAndPartials<T_y,T_low,T_high> 
        operands_and_partials(y,alpha,beta);
      for (size_t n = 0; n < N; n++) {
        if (include_summand<propto,T_low,T_high>::value)
          logp -= log_beta_minus_alpha[n];

        if (!is_constant_struct<T_low>::value)
          operands_and_partials.d_x2[n] += inv_beta_minus_alpha[n];
        if (!is_constant_struct<T_high>::value)
          operands_and_partials.d_x3[n] -= inv_beta_minus_alpha[n];
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_low, typename T_high>
    inline
    typename return_type<T_y,T_low,T_high>::type
    uniform_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      return uniform_log<false>(y,alpha,beta);
    }

    template <typename T_y, typename T_low, typename T_high>
    typename return_type<T_y,T_low,T_high>::type
    uniform_cdf(const T_y& y, const T_low& alpha, const T_high& beta) {
      static const char* function = "stan::prob::uniform_cdf(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_greater;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta)))
        return 1.0;

      // set up return value accumulator
      double cdf(1.0);
      if(!check_not_nan(function, y, "Random variable", &cdf))
        return cdf;
      if (!check_finite(function, alpha, "Lower bound parameter", &cdf))
        return cdf;
      if (!check_finite(function, beta, "Upper bound parameter", &cdf))
        return cdf;
      if (!check_greater(function, beta, alpha, "Upper bound parameter", &cdf))
        return cdf;
      if (!(check_consistent_sizes(function,
                                   y,alpha,beta,
                                   "Random variable","Lower bound parameter",
                                   "Upper bound parameter",
                                   &cdf)))
        return cdf;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_low> alpha_vec(alpha);
      VectorView<const T_high> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        if (y_dbl < value_of(alpha_vec[n]) 
            || y_dbl > value_of(beta_vec[n]))
          return 0.0;
      }
   
      agrad::OperandsAndPartials<T_y,T_low,T_high> 
        operands_and_partials(y,alpha,beta);
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
        const double b_min_a = beta_dbl - alpha_dbl;
        const double cdf_ = (y_dbl - alpha_dbl) / b_min_a;

        //cdf
        cdf *= cdf_;

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += 1.0 / b_min_a / cdf_;
        if (!is_constant_struct<T_low>::value)
          operands_and_partials.d_x2[n] += (y_dbl - beta_dbl) / b_min_a 
            / b_min_a / cdf_;
        if (!is_constant_struct<T_high>::value)
          operands_and_partials.d_x3[n] -= 1.0 / b_min_a;
      }

      if (!is_constant_struct<T_y>::value)
        for (size_t n = 0; n < stan::length(y); ++n) 
          operands_and_partials.d_x1[n] *= cdf;
      if (!is_constant_struct<T_low>::value)
        for (size_t n = 0; n < stan::length(alpha); ++n) 
          operands_and_partials.d_x2[n] *= cdf;
      if (!is_constant_struct<T_high>::value)
        for (size_t n = 0; n < stan::length(beta); ++n) 
          operands_and_partials.d_x3[n] *= cdf;

      return operands_and_partials.to_var(cdf);
    }
   
    template <typename T_y, typename T_low, typename T_high>
    typename return_type<T_y,T_low,T_high>::type
    uniform_cdf_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      static const char* function = "stan::prob::uniform_cdf_log(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_greater;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      double cdf_log(0.0);
      if(!check_not_nan(function, y, "Random variable", &cdf_log))
        return cdf_log;
      if (!check_finite(function, alpha, "Lower bound parameter", &cdf_log))
        return cdf_log;
      if (!check_finite(function, beta, "Upper bound parameter", &cdf_log))
        return cdf_log;
      if (!check_greater(function, beta, alpha, "Upper bound parameter", 
                         &cdf_log))
        return cdf_log;
      if (!(check_consistent_sizes(function,
                                   y,alpha,beta,
                                   "Random variable","Lower bound parameter",
                                   "Upper bound parameter",
                                   &cdf_log)))
        return cdf_log;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_low> alpha_vec(alpha);
      VectorView<const T_high> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);

      agrad::OperandsAndPartials<T_y,T_low,T_high> 
        operands_and_partials(y,alpha,beta);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        if (y_dbl < value_of(alpha_vec[n]) 
            || y_dbl > value_of(beta_vec[n]))
          return stan::math::negative_infinity();
        if (y_dbl == value_of(beta_vec[n]))
          return operands_and_partials.to_var(0.0);
      }

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
        const double b_min_a = beta_dbl - alpha_dbl;
        const double cdf_log_ = (y_dbl - alpha_dbl) / b_min_a;

        //cdf_log
        cdf_log += log(cdf_log_);

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += 1.0 / b_min_a / cdf_log_;
        if (!is_constant_struct<T_low>::value)
          operands_and_partials.d_x2[n] += (y_dbl - beta_dbl) / b_min_a
            / b_min_a / cdf_log_;
        if (!is_constant_struct<T_high>::value)
          operands_and_partials.d_x3[n] -= 1.0 / b_min_a;
      }

      return operands_and_partials.to_var(cdf_log);
    }

  template <typename T_y, typename T_low, typename T_high>
    typename return_type<T_y,T_low,T_high>::type
    uniform_ccdf_log(const T_y& y, const T_low& alpha, const T_high& beta) {
      static const char* function = "stan::prob::uniform_ccdf_log(%1%)";
      
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_greater;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta)))
        return 0.0;

      // set up return value accumulator
      double ccdf_log(0.0);
      if(!check_not_nan(function, y, "Random variable", &ccdf_log))
        return ccdf_log;
      if (!check_finite(function, alpha, "Lower bound parameter", &ccdf_log))
        return ccdf_log;
      if (!check_finite(function, beta, "Upper bound parameter", &ccdf_log))
        return ccdf_log;
      if (!check_greater(function, beta, alpha, "Upper bound parameter", 
                         &ccdf_log))
        return ccdf_log;
      if (!(check_consistent_sizes(function,
                                   y,alpha,beta,
                                   "Random variable","Lower bound parameter",
                                   "Upper bound parameter",
                                   &ccdf_log)))
        return ccdf_log;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_low> alpha_vec(alpha);
      VectorView<const T_high> beta_vec(beta);
      size_t N = max_size(y, alpha, beta);

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        if (y_dbl < value_of(alpha_vec[n]) 
            || y_dbl > value_of(beta_vec[n]))
          return 0.0;
        if (y_dbl == value_of(beta_vec[n]))
          return LOG_ZERO;
      }
   
      agrad::OperandsAndPartials<T_y,T_low,T_high> 
        operands_and_partials(y,alpha,beta);
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double alpha_dbl = value_of(alpha_vec[n]);
        const double beta_dbl = value_of(beta_vec[n]);
        const double b_min_a = beta_dbl - alpha_dbl;
        const double ccdf_log_ = 1.0 - (y_dbl - alpha_dbl) / b_min_a;

        //ccdf_log
        ccdf_log += log(ccdf_log_);

        //gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= 1.0 / b_min_a / ccdf_log_;
        if (!is_constant_struct<T_low>::value)
          operands_and_partials.d_x2[n] -= (y_dbl - beta_dbl) / b_min_a 
            / b_min_a / ccdf_log_;
        if (!is_constant_struct<T_high>::value)
          operands_and_partials.d_x3[n] += (y_dbl - alpha_dbl) / b_min_a 
            / b_min_a / ccdf_log_;
      }

      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    uniform_rng(const double alpha,
                const double beta,
                RNG& rng) {
      using boost::variate_generator;
      using boost::random::uniform_real_distribution;
      variate_generator<RNG&, uniform_real_distribution<> >
        uniform_rng(rng, uniform_real_distribution<>(alpha, beta));
      return uniform_rng();
    }
  }
}
#endif
