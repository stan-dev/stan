#ifndef STAN_MATH_PRIM_SCAL_PROB_SKEW_NORMAL_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_SKEW_NORMAL_RNG_HPP

#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/fun/owens_t.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/prob/uniform_rng.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline double
    skew_normal_rng(const double mu,
                    const double sigma,
                    const double alpha,
                    RNG& rng) {
      boost::math::skew_normal_distribution<> dist(mu, sigma, alpha);

      static const char* function("stan::math::skew_normal_rng");

      using stan::math::check_positive;
      using stan::math::check_finite;

      check_finite(function, "Location parameter", mu);
      check_finite(function, "Shape parameter", alpha);
      check_positive(function, "Scale parameter", sigma);

      return quantile(dist, stan::math::uniform_rng(0.0, 1.0, rng));
    }
  }
}
#endif

