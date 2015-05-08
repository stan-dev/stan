#ifndef STAN_MATH_PRIM_SCAL_PROB_UNIFORM_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_UNIFORM_RNG_HPP

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline double
    uniform_rng(const double alpha,
                const double beta,
                RNG& rng) {
      using boost::variate_generator;
      using boost::random::uniform_real_distribution;

      static const char* function("stan::math::uniform_rng");

      using stan::math::check_finite;
      using stan::math::check_greater;

      check_finite(function, "Lower bound parameter", alpha);
      check_finite(function, "Upper bound parameter", beta);
      check_greater(function, "Upper bound parameter", beta, alpha);

      variate_generator<RNG&, uniform_real_distribution<> >
        uniform_rng(rng, uniform_real_distribution<>(alpha, beta));
      return uniform_rng();
    }
  }
}
#endif
