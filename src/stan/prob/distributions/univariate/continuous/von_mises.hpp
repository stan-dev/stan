#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__VON_MISES_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__VON_MISES_HPP

#include <stan/agrad/partials_vari.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_greater.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_positive_finite.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>
#include <stan/prob/traits.hpp>

namespace stan { 
  
  namespace prob {

    template<bool propto,
             typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa) {
      static char const* const function = "stan::prob::von_mises_log";

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(kappa)))
        return 0.0;

      using stan::is_constant_struct;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_positive_finite;
      using stan::error_handling::check_greater;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_consistent_sizes;
      using stan::math::value_of;

      // Result accumulator.
      double logp = 0.0;

      // Validate arguments.
      check_finite(function, "Random variable", y);
      check_finite(function, "Location paramter", mu);
      check_positive_finite(function, "Scale parameter", kappa);
      check_consistent_sizes(function, 
                             "Random variable", y, 
                             "Location parameter", mu, 
                             "Scale parameter", kappa);

 
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value) 
        return logp;

      // Determine constants.
      const bool y_const = is_constant_struct<T_y>::value;
      const bool mu_const = is_constant_struct<T_loc>::value;
      const bool kappa_const = is_constant_struct<T_scale>::value;
      
      // Determine which expensive computations to perform.
      const bool compute_bessel0 = include_summand<propto,T_scale>::value;
      const bool compute_bessel1 = !kappa_const;
      const double TWO_PI = 2.0 * stan::math::pi();
      
      // Wrap scalars into vector views.
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> kappa_vec(kappa);

      DoubleVectorView<true,is_vector<T_scale>::value> kappa_dbl(length(kappa));
      DoubleVectorView<include_summand<propto,T_scale>::value,is_vector<T_scale>::value> log_bessel0(length(kappa));
      for (size_t i = 0; i < length(kappa); i++) {
        kappa_dbl[i] = value_of(kappa_vec[i]);
        if (include_summand<propto,T_scale>::value)
          log_bessel0[i] = log(boost::math::cyl_bessel_i(0, value_of(kappa_vec[i])));
      }
      agrad::OperandsAndPartials<T_y, T_loc, T_scale> oap(y, mu, kappa);

      size_t N = max_size(y, mu, kappa);

      for (size_t n = 0; n < N; n++) {
        // Extract argument values.
        const double y_ = value_of(y_vec[n]);
        const double y_dbl =  y_ - std::floor(y_ / TWO_PI) * TWO_PI;
        const double mu_dbl = value_of(mu_vec[n]);
        
        // Reusable values.
        double bessel0 = 0;
        if (compute_bessel0)
          bessel0 = boost::math::cyl_bessel_i(0, kappa_dbl[n]);
        double bessel1 = 0;
        if (compute_bessel1)
          bessel1 = boost::math::cyl_bessel_i(-1, kappa_dbl[n]);
        const double kappa_sin = kappa_dbl[n] * std::sin(mu_dbl - y_dbl);
        const double kappa_cos = kappa_dbl[n] * std::cos(mu_dbl - y_dbl);
        
        // Log probability.
        if (include_summand<propto>::value) 
          logp -= LOG_TWO_PI;
        if (include_summand<propto,T_scale>::value)
          logp -= log_bessel0[n];
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp += kappa_cos;
        
        // Gradient.
        if (!y_const) 
          oap.d_x1[n] += kappa_sin;
        if (!mu_const) 
          oap.d_x2[n] -= kappa_sin;
        if (!kappa_const) 
          oap.d_x3[n] += kappa_cos / kappa_dbl[n] - bessel1 / bessel0;
      }
      
      return oap.to_var(logp);
    }

    template<typename T_y, typename T_loc, typename T_scale>
    inline typename return_type<T_y,T_loc,T_scale>::type
    von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa) {
      return von_mises_log<false>(y, mu, kappa);
    }

    // The algorithm used in von_mises_rng is a modified version of the
    // algorithm in:
    //
    // Efficient Simulation of the von Mises Distribution
    // D. J. Best and N. I. Fisher
    // Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28, No. 2 (1979), pp. 152-157
    // 
    // See licenses/stan-license.txt for Stan license.

    template <class RNG>
    inline double
    von_mises_rng(const double mu,
                  const double kappa,
                  RNG& rng) {
      using boost::variate_generator;
      using stan::prob::uniform_rng;

      static const std::string function("stan::prob::von_mises_rng");

      stan::error_handling::check_finite(function, "mean", mu);
      stan::error_handling::check_positive_finite(function, "inverse of variance", kappa);

      double r = 1 + pow((1 + 4 * kappa * kappa), 0.5);
      double rho = 0.5 * (r - pow(2 * r, 0.5)) / kappa;
      double s = 0.5 * (1 + rho * rho) / rho;

      bool done = 0;
      double W;
      while (!done) {
        double Z = std::cos(stan::math::pi() * uniform_rng(0.0, 1.0, rng));
        W = (1 + s * Z) / (s + Z);
        double Y = kappa * (s - W);
        double U2 = uniform_rng(0.0, 1.0, rng);
        done = Y * (2 - Y) - U2 > 0;
        
        if(!done)
          done = log(Y / U2) + 1 - Y >= 0;
      }

      double U3 = uniform_rng(0.0, 1.0, rng) - 0.5;
      double sign = ((U3 >= 0) - (U3 <= 0));

      //  it's really an fmod() with a positivity constraint
      return sign * std::acos(W) + fmod(fmod(mu,2*stan::math::pi())+2*stan::math::pi(),2*stan::math::pi());
    }

  } 
}
#endif
