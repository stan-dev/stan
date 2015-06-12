#ifndef STAN_MATH_PRIM_SCAL_PROB_PARETO_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_PARETO_RNG_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>


namespace stan {
  namespace math {

    template <class RNG>
    inline double
    pareto_rng(const double y_min,
               const double alpha,
               RNG& rng) {
      using boost::variate_generator;
      using boost::exponential_distribution;

      static const char* function("stan::math::pareto_rng");

      using stan::math::check_positive_finite;

      check_positive_finite(function, "Scale parameter", y_min);
      check_positive_finite(function, "Shape parameter", alpha);

      variate_generator<RNG&, exponential_distribution<> >
        exp_rng(rng, exponential_distribution<>(alpha));
      return y_min * std::exp(exp_rng());
    }
  }
}
#endif
