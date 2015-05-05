#ifndef STAN_MATH_PRIM_SCAL_PROB_BETA_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_BETA_RNG_HPP

#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline double
    beta_rng(const double alpha,
             const double beta,
             RNG& rng) {
      using boost::variate_generator;
      using boost::random::gamma_distribution;
      // Error checks
      static const char* function("stan::math::beta_rng");

      using stan::math::check_positive_finite;

      check_positive_finite(function, "First shape parameter", alpha);
      check_positive_finite(function, "Second shape parameter", beta);

      variate_generator<RNG&, gamma_distribution<> >
        rng_gamma_alpha(rng, gamma_distribution<>(alpha, 1.0));
      variate_generator<RNG&, gamma_distribution<> >
        rng_gamma_beta(rng, gamma_distribution<>(beta, 1.0));
      double a = rng_gamma_alpha();
      double b = rng_gamma_beta();
      return a / (a + b);
    }

  }
}
#endif
