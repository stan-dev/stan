#ifndef STAN_MATH_PRIM_SCAL_PROB_PARETO_TYPE_2_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_PARETO_TYPE_2_RNG_HPP

#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/prob/uniform_rng.hpp>


namespace stan {
  namespace math {

    template <class RNG>
    inline double
    pareto_type_2_rng(const double mu,
                      const double lambda,
                      const double alpha,
                      RNG& rng) {
      static const char* function("stan::math::pareto_type_2_rng");

      stan::math::check_positive(function, "scale parameter", lambda);

      double uniform_01 = stan::math::uniform_rng(0.0, 1.0, rng);


      return (std::pow(1.0 - uniform_01, -1.0 / alpha) - 1.0) * lambda + mu;
    }
  }
}
#endif
