#ifndef STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_2_CDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_2_CDF_LOG_HPP

#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/prob/beta_cdf_log.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>
#include <vector>

namespace stan {

  namespace math {

    template <typename T_n, typename T_location,
              typename T_precision>
    typename return_type<T_location, T_precision>::type
    neg_binomial_2_cdf_log(const T_n& n,
                           const T_location& mu,
                           const T_precision& phi) {
      // Size checks
      if ( !( stan::length(n) && stan::length(mu)
              && stan::length(phi) ) )
        return 0.0;

      using stan::math::check_nonnegative;
      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::check_less;
      using std::log;
      using std::log;

      static const char* function("stan::math::neg_binomial_2_cdf");
      check_positive_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Precision parameter", phi);
      check_not_nan(function, "Random variable", n);
      check_consistent_sizes(function,
                             "Random variable", n,
                             "Location parameter", mu,
                             "Precision Parameter", phi);

      VectorView<const T_n> n_vec(n);
      VectorView<const T_location> mu_vec(mu);
      VectorView<const T_precision> phi_vec(phi);

      size_t size_phi_mu = max_size(mu, phi);
      size_t size_n = length(n);

      std::vector<typename return_type<T_location, T_precision>::type>
        phi_mu(size_phi_mu);
      std::vector<typename return_type<T_n>::type> np1(size_n);

      for (size_t i = 0; i < size_phi_mu; i++)
        phi_mu[i] = phi_vec[i]/(phi_vec[i]+mu_vec[i]);

      for (size_t i = 0; i < size_n; i++)
        if (n_vec[i] < 0)
          return log(0.0);
        else
          np1[i] = n_vec[i] + 1.0;

      if (size_n == 1) {
        if (size_phi_mu == 1)
          return beta_cdf_log(phi_mu[0], phi, np1[0]);
        else
          return beta_cdf_log(phi_mu, phi, np1[0]);
      } else {
        if (size_phi_mu == 1)
          return beta_cdf_log(phi_mu[0], phi, np1);
        else
          return beta_cdf_log(phi_mu, phi, np1);
      }
    }
  }
}
#endif
