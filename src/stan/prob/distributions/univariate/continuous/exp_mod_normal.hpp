#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__EXP__MOD__NORMAL__HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__EXP__MOD__NORMAL__HPP__

#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/math/functions/value_of.hpp>

namespace stan {

  namespace prob {

    template <bool propto, 
              typename T_y, typename T_loc, typename T_scale,typename T_inv_scale>
    typename return_type<T_y,T_loc,T_scale, T_inv_scale>::type
    exp_mod_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                       const T_inv_scale& lambda) {
      static const char* function = "stan::prob::exp_mod_normal_log(%1%)";

      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(lambda)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      if (!check_not_nan(function, y, "Random variable", &logp))
        return logp;
      if (!check_finite(function, mu, "Location parameter", 
                        &logp))
        return logp;
      if (!check_finite(function, lambda, "Inv_scale parameter", 
                        &logp))
        return logp;
      if (!check_positive(function, lambda, "Inv_scale parameter", 
                          &logp))
        return logp;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,lambda,
                                   "Random variable","Location parameter",
                                   "Scale parameter", "Inv_scale paramter",
                                   &logp)))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale,T_inv_scale>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale> 
        operands_and_partials(y, mu, sigma,lambda);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);

      for (size_t n = 0; n < N; n++) {
        //pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double sigma_dbl = value_of(sigma_vec[n]);
        const double lambda_dbl = value_of(lambda_vec[n]);

        const double pi_dbl = boost::math::constants::pi<double>();

        // log probability
        if (include_summand<propto>::value)
          logp -= log(2.0);
        if (include_summand<propto, T_inv_scale>::value)
          logp += log(lambda_dbl);
        if (include_summand<propto,T_y,T_loc,T_scale,T_inv_scale>::value)
          logp += lambda_dbl
            * (mu_dbl + 0.5 * lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) 
            + log(boost::math::erfc((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) 
                                    / (std::sqrt(2.0) * sigma_dbl)));

        // gradients
        const double deriv_logerfc 
          = -2.0 / std::sqrt(pi_dbl) 
          * exp(-(mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) 
                / (std::sqrt(2.0) * sigma_dbl) 
                * (mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) 
                / (sigma_dbl * std::sqrt(2.0))) 
          / boost::math::erfc((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl)
                              / (sigma_dbl * std::sqrt(2.0)));

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] 
            += -lambda_dbl 
            + deriv_logerfc * -1.0 / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n] 
            += lambda_dbl 
            + deriv_logerfc / (sigma_dbl * std::sqrt(2.0));
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n] 
            += sigma_dbl * lambda_dbl * lambda_dbl 
            + deriv_logerfc 
            * (-mu_dbl / (sigma_dbl * sigma_dbl * std::sqrt(2.0)) 
               + lambda_dbl / std::sqrt(2.0) 
               + y_dbl / (sigma_dbl * sigma_dbl * std::sqrt(2.0)));
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x4[n]
            += 1 / lambda_dbl + lambda_dbl * sigma_dbl * sigma_dbl 
            + mu_dbl - y_dbl + deriv_logerfc * sigma_dbl / std::sqrt(2.0);
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_inv_scale>
    inline
    typename return_type<T_y,T_loc,T_scale, T_inv_scale>::type
    exp_mod_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_inv_scale& lambda) {
      return exp_mod_normal_log<false>(y,mu,sigma,lambda);
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_inv_scale>
    typename return_type<T_y,T_loc,T_scale,T_inv_scale>::type
    exp_mod_normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_inv_scale& lambda) {
      static const char* function = "stan::prob::exp_mod_normal_cdf(%1%)";

      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;

      typename return_type<T_y, T_loc, T_scale, T_inv_scale>::type cdf(1);

      //check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(sigma)
            && stan::length(lambda)))
        return cdf;

      if (!check_not_nan(function, y, "Random variable", &cdf))
        return cdf;
      if (!check_finite(function, mu, "Location parameter", &cdf))
        return cdf;
      if (!check_not_nan(function, sigma, "Scale parameter", 
                         &cdf))
        return cdf;      
      if (!check_finite(function, sigma, "Scale parameter", &cdf))
        return cdf;
      if (!check_positive(function, sigma, "Scale parameter", 
                          &cdf))
        return cdf;
      if (!check_finite(function, lambda, "Inv_scale parameter", &cdf))
        return cdf;
      if (!check_positive(function, lambda, "Inv_scale parameter", 
                          &cdf))
        return cdf;
      if (!(check_consistent_sizes(function,
                                   y,mu,sigma,lambda,
                                   "Random variable","Location parameter",
                                   "Scale parameter","Inv_scale paramter",
                                   &cdf)))
        return cdf;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);
      
      for (size_t n = 0; n < N; n++) {
        if(boost::math::isinf(y_vec[n]))
          {
            if (y_vec[n] < 0.0)
              return cdf * 0.0;
          }
        cdf *= 0.5 * (1 + erf((y_vec[n] - mu_vec[n]) / (sqrt(2.0) * sigma_vec[n]))) 
          - exp(-lambda_vec[n] * (y_vec[n] - mu_vec[n]) 
                + lambda_vec[n] * sigma_vec[n] * lambda_vec[n] * sigma_vec[n] / 2.0) 
          * (0.5 * (1 + erf((y_vec[n] - mu_vec[n] - sigma_vec[n] * lambda_vec[n] * sigma_vec[n]) 
                            / (sqrt(2.0) * sigma_vec[n]))));
      }

      return cdf;
    }

    template <class RNG>
    inline double
    exp_mod_normal_rng(const double mu,
                       const double sigma,
                       const double lambda,
                       RNG& rng) {
      return stan::prob::normal_rng(mu, sigma,rng) + stan::prob::exponential_rng(lambda, rng);
    }
  }
}
#endif



