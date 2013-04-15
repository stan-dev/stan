#ifndef __STAN__PROB__DISTRIBUTIONS__DOUBLE_EXPONENTIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__DOUBLE_EXPONENTIAL_HPP__

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {
    template<typename T>
    inline int sign(const T& z) {
      return (z == 0) ? 0 : z < 0 ? -1 : 1;
    }

    // DoubleExponential(y|mu,sigma)  [sigma > 0]
    // FIXME: add documentation
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, 
                           const T_loc& mu, const T_scale& sigma) {
      static const char* function
        = "stan::prob::double_exponential_log(%1%)";
      
      using stan::is_constant_struct;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;
      using std::log;
      using std::fabs;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);
      if(!check_finite(function, y, "Random variable", &logp))
        return logp;
      if(!check_finite(function, mu, "Location parameter", 
                       &logp))
        return logp;
      if(!check_finite(function, sigma, "Scale parameter", 
                       &logp))
        return logp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,
                                   "Random variable","Location parameter","Shape parameter",
                                   &logp)))
        return logp;
      
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, mu, sigma);
      agrad::OperandsAndPartials<T_y,T_loc,T_scale> operands_and_partials(y, mu, sigma);

      DoubleVectorView<include_summand<propto,T_y,T_loc,T_scale>::value,is_vector<T_scale>::value> 
        inv_sigma(length(sigma));
      DoubleVectorView<!is_constant_struct<T_scale>::value,is_vector<T_scale>::value> 
        inv_sigma_squared(length(sigma));
      DoubleVectorView<include_summand<propto,T_scale>::value,is_vector<T_scale>::value> 
        log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        const double sigma_dbl = value_of(sigma_vec[i]);
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          inv_sigma[i] = 1.0 / sigma_dbl;
        if (include_summand<propto,T_scale>::value) 
          log_sigma[i] = log(value_of(sigma_vec[i]));
        if (!is_constant_struct<T_scale>::value) 
          inv_sigma_squared[i] = inv_sigma[i] * inv_sigma[i];
      }


      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
  
        // reusable subexpressions values
        const double y_m_mu = y_dbl - mu_dbl;
        const double fabs_y_m_mu = fabs(y_m_mu);

        // log probability
        if (include_summand<propto>::value)
          logp += NEG_LOG_TWO;
        if (include_summand<propto,T_scale>::value)
          logp -= log_sigma[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp -= fabs_y_m_mu * inv_sigma[n];
  
        // gradients
        double sign_y_m_mu_times_inv_sigma(0);
        if (!is_constant_struct<T_y>::value || !is_constant_struct<T_loc>::value)
          sign_y_m_mu_times_inv_sigma = sign(y_m_mu) * inv_sigma[n];
        if (!is_constant_struct<T_y>::value) {
          operands_and_partials.d_x1[n] -= sign_y_m_mu_times_inv_sigma;
        }
        if (!is_constant_struct<T_loc>::value) {
          operands_and_partials.d_x2[n] += sign_y_m_mu_times_inv_sigma;
        }
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] += -inv_sigma[n] + fabs_y_m_mu * inv_sigma_squared[n];
      }
      return operands_and_partials.to_var(logp);
    }


    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, const T_loc& mu, 
                           const T_scale& sigma) {
      return double_exponential_log<false>(y,mu,sigma);
    }

    /** 
     * Calculates the double exponential cumulative density function.
     *
     * \f$ f(y|\mu,\sigma) = \begin{cases} \
     \frac{1}{2} \exp\left(\frac{y-\mu}{\sigma}\right), \mbox{if } y < \mu \\ 
     1 - \frac{1}{2} \exp\left(-\frac{y-\mu}{\sigma}\right), \mbox{if } y \ge \mu \
     \end{cases}\f$
     * 
     * @param y A scalar variate.
     * @param mu The location parameter.
     * @param sigma The scale parameter.
     * 
     * @return The cumulative density function.
     */
    template <typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    double_exponential_cdf(const T_y& y, 
                           const T_loc& mu, const T_scale& sigma) {
      static const char* function
        = "stan::prob::double_exponential_cdf(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if(!check_finite(function, y, "Random variable", &lp))
        return lp;
      if(!check_finite(function, mu, "Location parameter", 
                       &lp))
        return lp;
      if(!check_finite(function, sigma, "Scale parameter", 
                       &lp))
        return lp;
      if(!check_positive(function, sigma, "Scale parameter", 
                         &lp))
        return lp;
      
      if (y < mu)
        return exp((y-mu)/sigma)/2;
      else
        return 1 - exp((mu-y)/sigma)/2;
    }
    
    template <class RNG>
    inline double
    double_exponential_rng(const double mu,
                           const double sigma,
                           RNG& rng) {
      using boost::variate_generator;
      using boost::random::uniform_01;
      using std::log;
      using std::abs;
      variate_generator<RNG&, uniform_01<> >
        rng_unit_01(rng, uniform_01<>());
      double a = 0;
      double laplaceRN = rng_unit_01();
      if(0.5 - laplaceRN > 0)
        a = 1.0;
      else if(0.5 - laplaceRN < 0)
        a = -1.0;
      return mu - sigma * a * log(1 - 2 * abs(0.5 - laplaceRN));
    }
  }
}
#endif
