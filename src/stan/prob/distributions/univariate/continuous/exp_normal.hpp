#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__EXP_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__EXP_NORMAL_HPP__

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // /**
    //  * <p>The result log probability is defined to be the sum of the
    //  * log probabilities for each observation/mean/deviation triple.
    //  * @param y (Sequence of) scalar(s).
    //  * @param mu (Sequence of) location parameter(s)
    //  * for the exp_normal distribution.
    //  * @param sigma (Sequence of) scale parameters for the exp_normal
    //  * distribution.
    //  * @return The log of the product of the densities.
    //  * @throw std::domain_error if the scale is not positive.
    //  * @tparam T_y Underlying type of scalar in sequence.
    //  * @tparam T_loc Type of location parameter.
    //  */
    // template <bool propto, 
    //           typename T_y, typename T_loc, typename T_scale,typename T_inv_scale,
    //           class Policy>
    // typename return_type<T_y,T_loc,T_scale, T_inv_scale>::type
    // exp_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
    // 			 const T_inv_scale& lambda, const Policy& /*policy*/) {
    //   static const char* function = "stan::prob::exp_normal_log(%1%)";

    //   using std::log;
    //   using stan::is_constant_struct;
    //   using stan::math::check_positive;
    //   using stan::math::check_finite;
    //   using stan::math::check_not_nan;
    //   using stan::math::check_consistent_sizes;
    //   using stan::math::value_of;
    //   using stan::prob::include_summand;

    //   // check if any vectors are zero length
    //   if (!(stan::length(y) 
    //         && stan::length(mu) 
    //         && stan::length(sigma)
    // 	    && stan::length(lambda)))
    //     return 0.0;

    //   // set up return value accumulator
    //   double logp(0.0);

    //   // validate args (here done over var, which should be OK)
    //   if (!check_not_nan(function, y, "Random variable", &logp, Policy()))
    //     return logp;
    //   if (!check_finite(function, mu, "Location parameter", 
    //                     &logp, Policy()))
    //     return logp;
    //   if (!check_finite(function, lambda, "Inv_scale parameter", 
    //                     &logp, Policy()))
    //     return logp;
    //   if (!check_positive(function, sigma, "Scale parameter", 
    //                       &logp, Policy()))
    //     return logp;
    //   if (!(check_consistent_sizes(function,
    //                                y,mu,sigma,lambda,
    //                                "Random variable","Location parameter","Scale parameter", "Inv_scale paramter",
    //                                &logp, Policy())))
    //     return logp;

    //   // check if no variables are involved and prop-to
    //   if (!include_summand<propto,T_y,T_loc,T_scale,T_inv_scale>::value)
    //     return 0.0;
      
    //   // set up template expressions wrapping scalars into vector views
    //   agrad::OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale> operands_and_partials(y, mu, sigma,lambda);

    //   VectorView<const T_y> y_vec(y);
    //   VectorView<const T_loc> mu_vec(mu);
    //   VectorView<const T_scale> sigma_vec(sigma);
    //   VectorView<const T_inv_scale> lambda_vec(lambda);
    //   size_t N = max_size(y, mu, sigma, lambda);

    //   DoubleVectorView<true,is_vector<T_scale>::value> inv_sigma(length(sigma));
    //   DoubleVectorView<include_summand<propto,T_scale>::value,is_vector<T_scale>::value> log_sigma(length(sigma));
    //   for (size_t i = 0; i < length(sigma); i++) {
    //     inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
    //     if (include_summand<propto,T_scale>::value)
    //       log_sigma[i] = log(value_of(sigma_vec[i]));
    //   }

    //   for (size_t n = 0; n < N; n++) {
    //     // pull out values of arguments
    //     const double y_dbl = value_of(y_vec[n]);
    //     const double mu_dbl = value_of(mu_vec[n]);
    // 	const double sigma_dbl = value_of(sigma_vec[n]);
    // 	const double lambda_dbl = value_of(lambda_vec[n]);

    // 	const double pi_dbl = boost::math::constants::pi<double>();

    //     // log probability
    //     if (include_summand<propto>::value)
    //       logp += log(lambda_dbl) - log(2);
    //     if (include_summand<propto,T_y,T_loc,T_scale,T_inv_scale>::value)
    //       logp += lambda_dbl / 2.0 * (2.0 * mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - 2.0 * y_dbl) + log(1.0 - boost::math::erf((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)));

    //     // gradients
    //     if (!is_constant_struct<T_y>::value)
    //       operands_and_partials.d_x1[n] += -lambda_dbl + (2.0 / std::sqrt(pi_dbl)) * std::exp((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)) * (-1.0 / (std::sqrt(2) * sigma_dbl)) / (1.0 - boost::math::erf((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)));
    //     if (!is_constant_struct<T_loc>::value)
    //       operands_and_partials.d_x2[n] += lambda_dbl + (2.0 / std::sqrt(pi_dbl)) * std::exp((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)) * (1.0 / (std::sqrt(2) * sigma_dbl)) / (1.0 - boost::math::erf((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)));
    //     if (!is_constant_struct<T_scale>::value)
    //       operands_and_partials.d_x3[n] += sigma_dbl * lambda_dbl * lambda_dbl + (2.0 / std::sqrt(pi_dbl)) * std::exp((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)) * (lambda_dbl / std::sqrt(2)) / (1.0 - boost::math::erf((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)));
    // 	if (!is_constant_struct<T_inv_scale>::value)
    //       operands_and_partials.d_x4[n] += 1 / lambda_dbl + lambda_dbl * sigma_dbl * sigma_dbl + (2.0 / std::sqrt(pi_dbl)) * std::exp((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)) * (sigma_dbl / std::sqrt(2)) / (1.0 - boost::math::erf((mu_dbl + lambda_dbl * sigma_dbl * sigma_dbl - y_dbl) / (std::sqrt(2) * sigma_dbl)));
    //   }
    //   return operands_and_partials.to_var(logp);
    // }


    // template <bool propto,
    //           typename T_y, typename T_loc, typename T_scale, typename T_inv_scale>
    // inline
    // typename return_type<T_y,T_loc,T_scale, T_inv_scale>::type
    // exp_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_inv_scale& lambda) {
    //   return exp_normal_log<propto>(y,mu,sigma,lambda,stan::math::default_policy());
    // }

    // template <typename T_y, typename T_loc, typename T_scale, typename T_inv_scale,
    //           class Policy>
    // inline
    // typename return_type<T_y,T_loc,T_scale, T_inv_scale>::type
    // exp_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_inv_scale& lambda, const Policy&) {
    //   return exp_normal_log<false>(y,mu,sigma,lambda,Policy());
    // }

    // template <typename T_y, typename T_loc, typename T_scale, typename T_inv_scale>
    // inline
    // typename return_type<T_y,T_loc,T_scale, T_inv_scale>::type
    // exp_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_inv_scale& lambda) {
    //   return exp_normal_log<false>(y,mu,sigma,lambda,stan::math::default_policy());
    // }

    // /**
    //  * Calculates the exp_normal cumulative distribution function for the given
    //  * variate, location, and scale.
    //  * 
    //  * \f$\Phi(x) = \frac{1}{\sqrt{2 \pi}} \int_{-\inf}^x e^{-t^2/2} dt\f$.
    //  * 
    //  * Errors are configured by policy.  All variables must be finite
    //  * and the scale must be strictly greater than zero.
    //  * 
    //  * @param y A scalar variate.
    //  * @param mu The location of the exp_normal distribution.
    //  * @param sigma The scale of the exp_normal distriubtion
    //  * @return The unit exp_normal cdf evaluated at the specified arguments.
    //  * @tparam T_y Type of y.
    //  * @tparam T_loc Type of mean parameter.
    //  * @tparam T_scale Type of standard deviation paramater.
    //  * @tparam Policy Error-handling policy.
    //  */
    // template <typename T_y, typename T_loc, typename T_scale, typename T_inv_scale,
    //           class Policy>
    // typename return_type<T_y,T_loc,T_scale,T_inv_scale>::type
    // exp_normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_inv_scale& lambda,
    //          const Policy&) {
    //   static const char* function = "stan::prob::exp_normal_cdf(%1%)";

    //   using stan::math::check_positive;
    //   using stan::math::check_finite;
    //   using stan::math::check_not_nan;
    //   using stan::math::check_consistent_sizes;


    //   typename return_type<T_y, T_loc, T_scale, T_inv_scale>::type cdf(1);
    //   // check if any vectors are zero length
    //   if (!(stan::length(y) 
    //         && stan::length(mu) 
    //         && stan::length(sigma)
    // 	    && stan::length(lambda)))
    //     return cdf;

    //   if (!check_not_nan(function, y, "Random variable", &cdf, Policy()))
    //     return cdf;
    //   if (!check_finite(function, mu, "Location parameter", &cdf, Policy()))
    //     return cdf;
    //   if (!check_not_nan(function, sigma, "Scale parameter", 
    //                      &cdf, Policy()))
    //     return cdf;
    //   if (!check_positive(function, sigma, "Scale parameter", 
    //                       &cdf, Policy()))
    //     return cdf;
    //   if (!check_finite(function, lambda, "Inv_scale parameter", &cdf, Policy()))
    //     return cdf;
    //   if (!check_positive(function, lambda, "Inv_scale parameter", 
    //                      &cdf, Policy()))
    //     return cdf;
    //   if (!(check_consistent_sizes(function,
    //                                y,mu,sigma,lambda,
    //                                "Random variable","Location parameter","Scale parameter","Inv_scale paramter",
    //                                &cdf, Policy())))
    //     return cdf;

    //   VectorView<const T_y> y_vec(y);
    //   VectorView<const T_loc> mu_vec(mu);
    //   VectorView<const T_scale> sigma_vec(sigma);
    //   VectorView<const T_inv_scale> lambda_vec(lambda);
    //   size_t N = max_size(y, mu, sigma, lambda);
      
    //   for (size_t n = 0; n < N; n++) {
    //     // reusable subexpression values
    // 	const double u_dbl = (lambda_vec[n]) * ((y_vec[n]) - (mu_vec[n]));
    // 	const double v_dbl = (lambda_vec[n]) * (sigma_vec[n]);

    //     cdf *= (0.5 * (1 + boost::math::erf(u_dbl / v_dbl))) * (lambda_vec[n]) - std::exp(-u_dbl + v_dbl * v_dbl / 2.0 + std::log((0.5 * (1 + boost::math::erf(u_dbl - v_dbl))) - (mu_vec[n])) * (lambda_vec[n]));
    //   }
    //   return cdf;
    // }

    // template <typename T_y, typename T_loc, typename T_scale, typename T_inv_scale>
    // inline
    // typename return_type<T_y, T_loc, T_scale, T_inv_scale>::type
    // exp_normal_cdf(const T_y& y, const T_loc& mu, const T_scale& sigma, const T_inv_scale& lambda) {
    //   return exp_normal_cdf(y,mu,sigma,lambda,stan::math::default_policy());
    // }


    template <class RNG>
    inline double
    exp_normal_rng(double mu,
		    double sigma,
		    double lambda,
		    RNG& rng) {
      return stan::prob::normal_rng(mu + stan::prob::exponential_rng(lambda, rng), sigma, rng);
    }
  }
}
#endif



