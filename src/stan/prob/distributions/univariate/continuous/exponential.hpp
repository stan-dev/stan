#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__EXPONENTIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__EXPONENTIAL_HPP__

#include <boost/random/exponential_distribution.hpp>
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
     * The log of an exponential density for y with the specified
     * inverse scale parameter.
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
       y 
       &\sim& 
       \mbox{\sf{Expon}}(\beta) \\
       \log (p (y \,|\, \beta) )
       &=& 
       \log \left( \beta \exp^{-\beta y} \right) \\
       &=& 
       \log (\beta) - \beta y \\
       & & 
       \mathrm{where} \; y > 0
     \f}
     *
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <bool propto, typename T_y, typename T_inv_scale>
    typename return_type<T_y,T_inv_scale>::type
    exponential_log(const T_y& y, const T_inv_scale& beta) {
      static const char* function = "stan::prob::exponential_log(%1%)";

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(beta)))
        return 0.0;
      
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      
      double logp(0.0);
      if(!check_not_nan(function, y, "Random variable", &logp))
        return logp;
      if(!check_finite(function, beta, "Inverse scale parameter", &logp))
        return logp;
      if(!check_positive(function, beta, "Inverse scale parameter", &logp))
        return logp;

      if (!(check_consistent_sizes(function,
                                   y,beta,
                                   "Random variable","Inverse scale parameter",
                                   &logp)))
        return logp;
      
      
      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, beta);
      
      DoubleVectorView<
        include_summand<propto,T_inv_scale>::value,
        is_vector<T_inv_scale>::value> log_beta(length(beta));
      for (size_t i = 0; i < length(beta); i++)
        if (include_summand<propto,T_inv_scale>::value)
          log_beta[i] = log(value_of(beta_vec[i]));

      agrad::OperandsAndPartials<T_y,T_inv_scale> operands_and_partials(y, beta);

      for (size_t n = 0; n < N; n++) {
        const double beta_dbl = value_of(beta_vec[n]);
        const double y_dbl = value_of(y_vec[n]);
        if (include_summand<propto,T_inv_scale>::value)
          logp += log_beta[n];
        if (include_summand<propto,T_y,T_inv_scale>::value)
          logp -= beta_dbl * y_dbl;
  
        if (!is_constant_struct<T_y>::value) 
          operands_and_partials.d_x1[n] -= beta_dbl;
        if (!is_constant_struct<T_inv_scale>::value) 
          operands_and_partials.d_x2[n] += 1 / beta_dbl - y_dbl;
      }
      return operands_and_partials.to_var(logp);
    }
    
    template <typename T_y, typename T_inv_scale>
    inline
    typename return_type<T_y,T_inv_scale>::type
    exponential_log(const T_y& y, const T_inv_scale& beta) {
      return exponential_log<false>(y,beta);
    }



    /**
     * Calculates the exponential cumulative distribution function for
     * the given y and beta.
     *
     * Inverse scale parameter must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     * @param y A scalar variable.
     * @param beta Inverse scale parameter.
     * @tparam T_y Type of scalar.
     * @tparam T_inv_scale Type of inverse scale.
     * @tparam Policy Error-handling policy.
     */
    template <typename T_y, typename T_inv_scale>
    typename return_type<T_y,T_inv_scale>::type
    exponential_cdf(const T_y& y, const T_inv_scale& beta) {

      static const char* function = "stan::prob::exponential_cdf(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

     double cdf(1.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(beta)))
        return cdf;

      if(!check_not_nan(function, y, "Random variable", &cdf))
        return cdf;
      if(!check_nonnegative(function, y, "Random variable", &cdf))
        return cdf;
      if(!check_finite(function, beta, "Inverse scale parameter", &cdf))
        return cdf;
      if(!check_positive(function, beta, "Inverse scale parameter", &cdf))
        return cdf;

      agrad::OperandsAndPartials<T_y, T_inv_scale> 
        operands_and_partials(y, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, beta);
      for (size_t n = 0; n < N; n++) {   
        const double beta_dbl = value_of(beta_vec[n]);     
        const double y_dbl = value_of(y_vec[n]);     
        const double one_m_exp = 1.0 - exp(-beta_dbl * y_dbl);

        // cdf
        cdf *= one_m_exp;
      }

      for(size_t n = 0; n < N; n++) {
        const double beta_dbl = value_of(beta_vec[n]);     
        const double y_dbl = value_of(y_vec[n]);     
        const double one_m_exp = 1.0 - exp(-beta_dbl * y_dbl);

        // gradients
        double rep_deriv = exp(-beta_dbl * y_dbl) / one_m_exp;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += rep_deriv * beta_dbl * cdf;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[n] += rep_deriv * y_dbl * cdf;
      }

      return operands_and_partials.to_var(cdf);
    }

    template <typename T_y, typename T_inv_scale>
    typename return_type<T_y,T_inv_scale>::type
    exponential_cdf_log(const T_y& y, const T_inv_scale& beta) {

      static const char* function = "stan::prob::exponential_cdf_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

     double cdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(beta)))
        return cdf_log;

      if(!check_not_nan(function, y, "Random variable", &cdf_log))
        return cdf_log;
      if(!check_nonnegative(function, y, "Random variable", &cdf_log))
        return cdf_log;
      if(!check_finite(function, beta, "Inverse scale parameter", &cdf_log))
        return cdf_log;
      if(!check_positive(function, beta, "Inverse scale parameter", &cdf_log))
        return cdf_log;

      agrad::OperandsAndPartials<T_y, T_inv_scale> 
        operands_and_partials(y, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, beta);
      for (size_t n = 0; n < N; n++) { 
        const double beta_dbl = value_of(beta_vec[n]);     
        const double y_dbl = value_of(y_vec[n]);            
        double one_m_exp = 1.0 - exp(-beta_dbl * y_dbl);
        // log cdf
        cdf_log += log(one_m_exp);

        // gradients
        double rep_deriv = -exp(-beta_dbl * y_dbl) / one_m_exp;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= rep_deriv * beta_dbl;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[n] -= rep_deriv * y_dbl;
      }
      return operands_and_partials.to_var(cdf_log);
    }

   template <typename T_y, typename T_inv_scale>
    typename return_type<T_y,T_inv_scale>::type
    exponential_ccdf_log(const T_y& y, const T_inv_scale& beta) {

      static const char* function = "stan::prob::exponential_ccdf_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

     double ccdf_log(0.0);
      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(beta)))
        return ccdf_log;

      if(!check_not_nan(function, y, "Random variable", &ccdf_log))
        return ccdf_log;
      if(!check_nonnegative(function, y, "Random variable", &ccdf_log))
        return ccdf_log;
      if(!check_finite(function, beta, "Inverse scale parameter", &ccdf_log))
        return ccdf_log;
      if(!check_positive(function, beta, "Inverse scale parameter", &ccdf_log))
        return ccdf_log;

      agrad::OperandsAndPartials<T_y, T_inv_scale> 
        operands_and_partials(y, beta);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_inv_scale> beta_vec(beta);
      size_t N = max_size(y, beta);
      for (size_t n = 0; n < N; n++) { 
        const double beta_dbl = value_of(beta_vec[n]);     
        const double y_dbl = value_of(y_vec[n]);            
        // log ccdf
        ccdf_log += -beta_dbl * y_dbl;

        // gradients
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= beta_dbl;
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x2[n] -= y_dbl;
      }
      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    exponential_rng(const double beta,
                    RNG& rng) {
      using boost::variate_generator;
      using boost::exponential_distribution;
      variate_generator<RNG&, exponential_distribution<> >
        exp_rng(rng, exponential_distribution<>(beta));
      return exp_rng();
    }
  }
}

#endif
