#ifndef STAN_MATH_PRIM_SCAL_PROB_POISSON_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_POISSON_RNG_HPP

#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>
#include <limits>

namespace stan {

  namespace math {

    template <class RNG>
    inline int
    poisson_rng(const double lambda,
                RNG& rng) {
      using boost::variate_generator;
      using boost::random::poisson_distribution;

      static const char* function("stan::math::poisson_rng");

      check_not_nan(function, "Rate parameter", lambda);
      check_nonnegative(function, "Rate parameter", lambda);
      check_less(function, "Rate parameter", lambda, POISSON_MAX_RATE);

      variate_generator<RNG&, poisson_distribution<> >
        poisson_rng(rng, poisson_distribution<>(lambda));
      return poisson_rng();
    }
  }
}
#endif
