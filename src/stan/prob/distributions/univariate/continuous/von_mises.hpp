#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__VON_MISES_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__VON_MISES_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan { namespace prob {

template<bool propto,
         typename T_y, typename T_loc, typename T_scale,
         typename Policy>
typename return_type<T_y,T_loc,T_scale>::type
von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa,
              Policy const& policy) {
  static char const* const function = "stan::prob::von_mises_log(%1%)";

  using stan::is_constant_struct;
  using stan::math::check_finite;
  using stan::math::check_positive;
  using stan::math::check_greater;
  using stan::math::check_less;
  using stan::math::check_consistent_sizes;
  using stan::math::value_of;

  // Result accumulator.
  typename return_type<T_y,T_loc,T_scale>::type logp = 0;

  double const pi = boost::math::constants::pi<double>();

  // Validate arguments.
  if (!check_finite(function, y, "Random variable", &logp, policy) ||
      !check_less(function, y, pi, "Random variable", &logp, policy) ||
      !check_greater(function, y, -pi, "Random variable", &logp, policy) ||
      !check_finite(function, mu, "Location paramter", &logp, policy) ||
      !check_less(function, mu, pi, "Location paramter", &logp, policy) ||
      !check_greater(function, mu, -pi, "Location paramter", &logp, policy) ||
      !check_finite(function, kappa, "Scale parameter", &logp, policy) ||
      !check_positive(function, kappa, "Scale parameter", &logp, policy) ||
      !check_consistent_sizes(function, y, mu, kappa, "Random variable",
                              "Location parameter", "Scale parameter",
                              &logp, policy))
    return logp;
 
  // Nothing to do.
  if (!include_summand<propto,T_y,T_loc,T_scale>::value) return logp;

  // Determine constants.
  bool const y_const = is_constant_struct<T_y>::value;
  bool const mu_const = is_constant_struct<T_loc>::value;
  bool const kappa_const = is_constant_struct<T_scale>::value;

  // Determine which expensive computations to perform.
  bool const compute_bessel0 = include_summand<propto,T_scale>::value || !kappa_const;
  bool const compute_bessel1 = !kappa_const;
  bool const compute_kappa_sin = !y_const || !kappa_const;

  // Wrap scalars into vector views.
  VectorView<T_y const> y_vec(y);
  VectorView<T_loc const> mu_vec(mu);
  VectorView<T_scale const> kappa_vec(kappa);
  typedef agrad::OperandsAndPartials<T_y,T_loc,T_scale> OAP;
  OAP oap(y, mu, kappa, y_vec, mu_vec, kappa_vec);

  for (size_t n = 0, N = max_size(y, mu, kappa); n != N; ++n) {
    // Extract argument values.
    double const y_dbl = value_of(y_vec[n]);
    double const mu_dbl = value_of(mu_vec[n]);
    double const kappa_dbl = value_of(kappa_vec[n]);

    // Reusable values.
    double const bessel0 = compute_bessel0 ? boost::math::cyl_bessel_i(0, kappa_dbl) : 0;
    double const bessel1 = compute_bessel1 ? boost::math::cyl_bessel_i(1, kappa_dbl) : 0;
    double const kappa_sin = compute_kappa_sin ? kappa_dbl * std::sin(mu_dbl - y_dbl) : 0;
    double const kappa_cos = kappa_dbl * std::cos(mu_dbl - y_dbl);

    // Log probability.
    if (include_summand<propto>::value) logp -= LOG_TWO_PI;
    if (include_summand<propto,T_scale>::value) logp -= std::log(bessel0);
    /*if (include_summand<propto,T_y,T_loc,T_scale>::value)*/ logp += kappa_cos;

    // Gradient.
    if (!y_const) oap.d_x1[n] += kappa_sin;
    if (!mu_const) oap.d_x2[n] -= kappa_sin;
    if (!kappa_const) oap.d_x3[n] += kappa_cos / kappa_dbl - bessel1 / bessel0;
  }

  return oap.to_var(logp);
}

template<bool propto, typename T_y, typename T_loc, typename T_scale>
inline typename return_type<T_y,T_loc,T_scale>::type
von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa) {
  return von_mises_log<propto>(y, mu, kappa, math::default_policy());
}

template<typename T_y, typename T_loc, typename T_scale, typename Policy>
inline typename return_type<T_y,T_loc,T_scale>::type
von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa,
              Policy const& policy) {
  return von_mises_log<false>(y, mu, kappa, policy);
}

template<typename T_y, typename T_loc, typename T_scale>
inline typename return_type<T_y,T_loc,T_scale>::type
von_mises_log(T_y const& y, T_loc const& mu, T_scale const& kappa) {
  return von_mises_log<false>(y, mu, kappa, math::default_policy());
}

} } // namespace prob stan

#endif // __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__VON_MISES_HPP__
