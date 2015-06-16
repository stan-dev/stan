#ifndef STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_NEG_BINOMIAL_RNG_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/prob/gamma_rng.hpp>
#include <stan/math/prim/scal/prob/poisson_rng.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline int
    neg_binomial_rng(const double alpha,
                     const double beta,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;
      using boost::random::poisson_distribution;
      using boost::gamma_distribution;

      static const char* function("stan::math::neg_binomial_rng");

      // gamma_rng params must be positive and finite
      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Inverse scale parameter", beta);

      double rng_from_gamma =
        variate_generator<RNG&, gamma_distribution<> >
        (rng, gamma_distribution<>(alpha, 1.0 / beta))();

      // same as the constraints for poisson_rng
      check_less(function,
        "Random number that came from gamma distribution",
        rng_from_gamma, POISSON_MAX_RATE);
      check_not_nan(function,
        "Random number that came from gamma distribution",
        rng_from_gamma);
      check_nonnegative(function,
        "Random number that came from gamma distribution",
        rng_from_gamma);

      return variate_generator<RNG&, poisson_distribution<> >
        (rng, poisson_distribution<>(rng_from_gamma))();
    }
  }
}
#endif
