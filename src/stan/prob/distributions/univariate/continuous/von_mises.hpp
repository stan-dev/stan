#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__VON_MISES_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__VON_MISES_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan { 
  
  namespace prob {

    template<bool propto,
             typename T_y, typename T_loc, typename T_scale>
    typename return_type<T_y,T_loc,T_scale>::type
    von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa) {
      static char const* const function = "stan::prob::von_mises_log(%1%)";

      // check if any vectors are zero length
      if (!(stan::length(y) 
            && stan::length(mu) 
            && stan::length(kappa)))
        return 0.0;

      using stan::is_constant_struct;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_greater;
      using stan::math::check_less;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;

      // Result accumulator.
      double logp = 0.0;

      double const pi = boost::math::constants::pi<double>();

      // Validate arguments.
      if (!check_finite(function, y, "Random variable", &logp))
        return logp;
      if(!check_less(function, y, pi, "Random variable", &logp))
        return logp;
      if(!check_greater(function, y, -pi, "Random variable", &logp))
        return logp;
      if(!check_finite(function, mu, "Location paramter", &logp))
        return logp;
      if(!check_less(function, mu, pi, "Location paramter", &logp))
        return logp;            
      if(!check_greater(function, mu, -pi, "Location paramter", &logp))
        return logp;  
      if(!check_finite(function, kappa, "Scale parameter", &logp))
        return logp;
      if(!check_positive(function, kappa, "Scale parameter", &logp))
        return logp;
      if(!check_consistent_sizes(function, y, mu, kappa, "Random variable",
                                  "Location parameter", "Scale parameter",
                                  &logp))
        return logp;
 
      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value) 
        return logp;

      // Determine constants.
      const bool y_const = is_constant_struct<T_y>::value;
      const bool mu_const = is_constant_struct<T_loc>::value;
      const bool kappa_const = is_constant_struct<T_scale>::value;
      
      // Determine which expensive computations to perform.
      const bool compute_bessel0 = include_summand<propto,T_scale>::value ||
        !kappa_const;
      const bool compute_bessel1 = !kappa_const;
      
      // Wrap scalars into vector views.
      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> kappa_vec(kappa);

      agrad::OperandsAndPartials<T_y, T_loc, T_scale> oap(y, mu, kappa);

      size_t N = max_size(y, mu, kappa);

      for (size_t n = 0; n < N; n++) {
        // Extract argument values.
        const double y_dbl = value_of(y_vec[n]);
        const double mu_dbl = value_of(mu_vec[n]);
        const double kappa_dbl = value_of(kappa_vec[n]);
        
        // Reusable values.
        const double bessel0 = compute_bessel0 ?
          boost::math::cyl_bessel_i(0, kappa_dbl) : 0;
        const double bessel1 = compute_bessel1 ?
          boost::math::cyl_bessel_i(-1, kappa_dbl) : 0;
        const double kappa_sin = kappa_dbl * std::sin(mu_dbl - y_dbl);
        const double kappa_cos = kappa_dbl * std::cos(mu_dbl - y_dbl);
        
        // Log probability.
        if (include_summand<propto>::value) 
          logp -= LOG_TWO_PI;
        if (include_summand<propto,T_scale>::value)
          logp -= log(bessel0);
        if (include_summand<propto,T_y,T_loc,T_scale>::value)
          logp += kappa_cos;
        
        // Gradient.
        if (!y_const) 
          oap.d_x1[n] += kappa_sin;
        if (!mu_const) 
          oap.d_x2[n] -= kappa_sin;
        if (!kappa_const) 
          oap.d_x3[n] += kappa_cos / kappa_dbl - bessel1 / bessel0;
      }
      
      return oap.to_var(logp);
    }

    template<typename T_y, typename T_loc, typename T_scale>
    inline typename return_type<T_y,T_loc,T_scale>::type
    von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa) {
      return von_mises_log<false>(y, mu, kappa);
    }
  } 
}
#endif
