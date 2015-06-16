#ifndef STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_2_CCDF_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_2_CCDF_LOG_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
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
#include <stan/math/prim/scal/prob/neg_binomial_ccdf_log.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <vector>

namespace stan {

  namespace math {

    // Temporary neg_binomial_2_ccdf implementation that
    // transforms the input parameters and calls neg_binomial_ccdf
    template <typename T_n, typename T_location, typename T_precision>
    typename return_type<T_location, T_precision>::type
    neg_binomial_2_ccdf_log(const T_n& n,
                            const T_location& mu,
                            const T_precision& phi) {
      if ( !( stan::length(n) && stan::length(mu) && stan::length(phi) ) )
        return 0.0;

      using stan::math::check_nonnegative;
      using stan::math::check_positive_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::check_less;

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

      size_t size_beta = max_size(mu, phi);

      std::vector<typename return_type<T_location, T_precision>::type>
        beta_vec(size_beta);
      for (size_t i = 0; i < size_beta; ++i)
        beta_vec[i] = phi_vec[i] / mu_vec[i];

      // Cast a vector of size 1 down to a
      // scalar to avoid dimension mismatch
      if (size_beta == 1)
        return neg_binomial_ccdf_log(n, phi, beta_vec[0]);
      else
        return neg_binomial_ccdf_log(n, phi, beta_vec);
    }
  }
}
#endif
