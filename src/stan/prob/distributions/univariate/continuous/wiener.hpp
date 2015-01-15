// Copyright (c) 2013, Joachim Vandekerckhove.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
//   * Neither the name of the University of California, Irvine nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__WIENER_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__WIENER_HPP__

#include <stan/math/functions/constants.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/prob/traits.hpp>
#include <boost/math/distributions.hpp>
#include <cmath>
#include <stan/agrad.hpp>
#include <stan/math/functions/value_of.hpp>




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
                 MY_LN_SQRT_PI = std::log(std::sqrt(boost::math::constants::pi<double>())),
                 WIENER_ERR = 0.000001,
                 LOG_ZERO = stan::math::negative_infinity();

    template <bool propto,
    typename T_y, typename T_alpha, typename T_tau, typename T_beta, typename T_delta>
    typename return_type<T_y,T_alpha,T_tau,T_beta,T_delta>::type
    wiener_log(const T_y& y, const T_alpha& alpha, const T_tau& tau,
               const T_beta& beta, const T_delta& delta) {
      static const std::string function("stan::prob::wiener_log(%1%)");

      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_consistent_sizes;
      
      using stan::math::value_of;
      using boost::math::tools::promote_args;
      using boost::math::isinf;
      using boost::math::isfinite;

      if (!(stan::length(y) 
            && stan::length(alpha) 
            && stan::length(beta) 
            && stan::length(tau) 
            && stan::length(delta)))
        return 0.0;

      typedef typename return_type<T_y,T_alpha,T_tau,T_beta,T_delta>::type T_return_type;
      T_return_type lp(0.0);

      check_not_nan (function, "Random variable", y);
      check_not_nan (function, "Boundary separation", alpha);
      check_not_nan (function, "A-priori bias", beta);
      check_not_nan (function, "Nondecision time", tau);
      check_not_nan (function, "Drift rate", delta);
      check_finite  (function, "Boundary separation", alpha);
      check_finite  (function, "A-priori bias", beta);
      check_finite  (function, "Nondecision time", tau);
      check_finite  (function, "Drift rate", delta);
      check_positive(function, "Random variable", y);
      check_positive(function, "Boundary separation", alpha);
      check_positive(function, "Nondecision time", tau);
      check_bounded (function, "A-priori bias", beta , 0, 1);
      check_consistent_sizes(function,
                             "Random variable", y,
                             "Boundary separation", alpha,
                             "A-priori bias", beta,
                             "Nondecision time", tau,
                             "Drift rate", delta);
      
      size_t N = std::max(max_size(y, alpha, beta), max_size(tau, delta));
      if (!N)
        return 0.0;
      VectorView<const T_y> y_vec(y);
      VectorView<const T_alpha> alpha_vec(alpha);
      VectorView<const T_beta> beta_vec(beta);
      VectorView<const T_tau> tau_vec(tau);
      VectorView<const T_delta> delta_vec(delta);

      if (!include_summand<propto,T_y,T_alpha,T_tau,T_beta,T_delta>::value) {
        return 0;
      }
      
      for (size_t i = 0; i < N; i++)
        if (y_vec[i] < tau_vec[i]) {
          lp = LOG_ZERO;
          return lp;
        }
      
      for (size_t i = 0; i < N; i++) {
        typename scalar_type<T_beta>::type one_minus_beta = 1.0 - beta_vec[i];
        typename scalar_type<T_alpha>::type alpha2 = alpha_vec[i] * alpha_vec[i];
        T_return_type x = y_vec[i];
        T_return_type kl, ks, tmp = 0;
        T_return_type k, K;


        x = x - tau_vec[i]; // remove non-decision time from x
        x = x / alpha2; // convert t to normalized time tt

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
            tmp += (one_minus_beta + 2.0 * k) * exp(-(pow((one_minus_beta + 2.0 * k), 2)) / 2.0 / x); // increment sum
          tmp = log(tmp) - 0.5 * log(2.0) - MY_LN_SQRT_PI - 1.5 * log(x); // add constant term
        }
        else { // if large t is better...
          K = ceil(kl); // round to smallest integer meeting error
          for (k = 1; k <= K; k++)
            tmp += k * exp(-(pow(k, 2)) * (pow(MY_PI, 2)) * x / 2.0) * sin(k * MY_PI * one_minus_beta); // increment sum
          tmp = log(tmp) + 2.0 * MY_LN_SQRT_PI; // add constant term
        }

        // convert to f(t|v,a,w) and return result
        lp += delta_vec[i] * alpha_vec[i] * one_minus_beta - pow(delta_vec[i], 2) * x * alpha2 / 2.0 - log(alpha2) + tmp;
      }
      
      return lp;
    }

    template <typename T_y, typename T_alpha, typename T_tau, typename T_beta, typename T_delta>
    inline
    typename return_type<T_y,T_alpha,T_tau,T_beta,T_delta>::type
    wiener_log(const T_y& y, const T_alpha& alpha, const T_tau& tau, const T_beta& beta, const T_delta& delta) {
      return wiener_log<false>(y,alpha,tau,beta,delta);
    }
    
    /* Not actually implemented yet
    template <class RNG>
    inline double
    wiener_rng(const double alpha,
         const double tau,
         const double beta,
         const double delta,
                     RNG& rng) {
      using boost::variate_generator;
      double a = tau+alpha*beta/stan::prob::normal_rng(delta,1,rng); 
      //stan::prob::wiener_rng(alpha, tau, beta, delta, rng);
      return a;
    }
    */
  }
}
#endif
