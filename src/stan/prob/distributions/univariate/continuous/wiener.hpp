#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__WIENER_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__WIENER_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/traits.hpp>
#include <boost/math/distributions.hpp>
#include <cmath>
#include <stan/agrad.hpp>
#include <stan/math/functions/value_of.hpp>

#define WIENER_ERR      0.000001

using std::vector;
using std::log;
using std::min;
using std::max;
using std::string;

namespace stan {

  namespace prob {
    /**
     * The log of the first passage time density function for a (Wiener) drift diffusion model for the given $y$,
     * boundary separation $\alpha$, nondecision time $\tau$, relative bias $\beta$, and drift rate $\delta$.
     * $\alpha$ and $\tau$ must be greater than 0, and $\beta$ must be between 0 and 1. $y$ should contain
     * reaction times in seconds, with upper-boundary responses strictly positive and lower-boundary response
     * times coded as strictly negative numbers.
     *
     * @param y A scalar variate.
     * @param alpha The boundary separation.
     * @param tau The nondecision time.
     * @param beta The relative bias.
     * @param delta The drift rate.
     * @return The log of the Wiener first passage time density of the specified arguments.
     */

    const double MY_PI = boost::math::constants::pi<double>(),
                 MY_LN_SQRT_PI = std::log(std::sqrt(boost::math::constants::pi<double>()));

    template <bool propto,
    typename T_y, typename T_alpha, typename T_tau, typename T_beta, typename T_delta>
    typename boost::math::tools::promote_args<T_y,T_alpha,T_tau,T_beta,T_delta>::type
    wiener_log(const T_y& y, const T_alpha& alpha, const T_tau& tau,
               const T_beta& beta, const T_delta& delta) {
      static const char* function = "stan::prob::wiener_log(%1%)";

      using stan::math::check_greater;
      using stan::math::check_not_nan;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_bounded;
      using stan::math::value_of;
      using boost::math::tools::promote_args;
      using boost::math::isinf;
      using boost::math::isfinite;

      typename promote_args<T_y,T_alpha,T_tau,T_beta,T_delta>::type lp(0.0);

      if (!check_not_nan (function, alpha, "Boundary separation", &lp))  return lp;
      if (!check_not_nan (function, beta , "A-priori bias"      , &lp))  return lp;
      if (!check_not_nan (function, tau  , "Nondecision time"   , &lp))  return lp;
      if (!check_not_nan (function, delta, "Drift rate"         , &lp))  return lp;
      if (!check_finite  (function, alpha, "Boundary separation", &lp))  return lp;
      if (!check_finite  (function, beta , "A-priori bias"      , &lp))  return lp;
      if (!check_finite  (function, tau  , "Nondecision time"   , &lp))  return lp;
      if (!check_finite  (function, delta, "Drift rate"         , &lp))  return lp;
      if (!check_positive(function, alpha, "Boundary separation", &lp))  return lp;
      if (!check_positive(function, tau  , "Nondecision time"   , &lp))  return lp;
      if (!check_bounded (function, beta , 0, 1, "A-priori bias", &lp))  return lp;

      if (fabs(y) < tau) {
        lp = LOG_ZERO;
        return lp;
      }

      double v = value_of(delta), w = value_of(beta), a = value_of(alpha), t = value_of(tau);
      double kl, ks, tmp = 0, x = value_of(y), a2 = a * a;
      int k, K;

      // extract RT and accuracy from x
      if (x < 0.0)  // error response
        x = fabs(x);
      else {      // correct response
        w = 1.0 - w;
        v = -v;
      }

      x = x - t; // remove non-decision time from x
      x = x / a2; // convert t to normalized time tt

      // calculate number of terms needed for large t
      if (MY_PI * x * WIENER_ERR < 1) { // if error threshold is set low enough
        kl = sqrt(-2.0 * log(MY_PI * x * WIENER_ERR) / (pow(MY_PI, 2) * x)); // bound
        kl = (kl > 1.0 / (MY_PI * sqrt(x))) ? kl : 1.0 / (MY_PI * sqrt(x)); // ensure boundary conditions met
      }
      else // if error threshold set too high
        kl = 1.0 / (MY_PI * sqrt(x)); // set to boundary condition

      // calculate number of terms needed for small t
      if ((2.0 * sqrt(2.0 * MY_PI * x) * WIENER_ERR) < 1) { // if error threshold is set low enough
        ks = 2.0 + sqrt(-2.0 * x * log(2.0 * sqrt(2.0 * MY_PI * x) * WIENER_ERR)); // bound
        ks = (ks > sqrt(x) + 1.0) ? ks : sqrt(x) + 1.0; // ensure boundary conditions are met
      }
      else // if error threshold was set too high
        ks = 2.0; // minimal kappa for that case

      // compute density: f(tt|0,1,w)
      if (ks < kl) { // if small t is better (i.e., lambda<0)
        K = ceil(ks); // round to smallest integer meeting error
        for (k = -floor((K - 1.0) / 2.0); k <= ceil((K - 1.0) / 2.0); k++)
          tmp += (w + 2.0 * k) * exp(-(pow((w + 2.0 * k), 2)) / 2.0 / x); // increment sum
        tmp = log(tmp) - 0.5 * log(2.0) - MY_LN_SQRT_PI - 1.5 * log(x); // add constant term
      }
      else { // if large t is better...
        K = ceil(kl); // round to smallest integer meeting error
        for (k = 1; k <= K; k++)
          tmp += k * exp(-(pow(k, 2)) * (pow(MY_PI, 2)) * x / 2.0) * sin(k * MY_PI * w); // increment sum
        tmp = log(tmp) + 2.0 * MY_LN_SQRT_PI; // add constant term
      }

      // convert to f(t|v,a,w) and return result
      lp = tmp + ((-v * a * w -(pow(v, 2)) * (x * a2) / 2.0) - log(a2));

      return lp;
    }

    template <typename T_y, typename T_alpha, typename T_tau, typename T_beta, typename T_delta>
    inline
    typename boost::math::tools::promote_args<T_y,T_alpha,T_tau,T_beta,T_delta>::type
    wiener_log(const T_y& y, const T_alpha& alpha, const T_tau& tau, const T_beta& beta, const T_delta& delta) {
      return wiener_log<false>(y,alpha,tau,beta,delta);
    }

    template <class RNG>
    inline double
    wiener_rng(const double alpha,
         const double tau,
         const double beta,
         const double delta,
                     RNG& rng) {
      using boost::variate_generator;
      double a = tau+alpha*beta/stan::prob::normal_rng(delta,1,rng); //not actually implemented yet
      //stan::prob::wiener_rng(alpha, tau, beta, delta, rng);
      return a;
    }
  }
}
#endif
