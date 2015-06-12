#ifndef STAN_MATH_PRIM_SCAL_PROB_LOGNORMAL_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_LOGNORMAL_RNG_HPP

#include <boost/random/lognormal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {
  namespace math {

    template <class RNG>
    inline double
    lognormal_rng(const double mu,
                  const double sigma,
                  RNG& rng) {
      using boost::variate_generator;
      using boost::random::lognormal_distribution;

      static const char* function("stan::math::lognormal_rng");

      using stan::math::check_finite;
      using stan::math::check_positive_finite;

      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      variate_generator<RNG&, lognormal_distribution<> >
        lognorm_rng(rng, lognormal_distribution<>(mu, sigma));
      return lognorm_rng();
    }
  }
}
#endif
