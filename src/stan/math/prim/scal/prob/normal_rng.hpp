#ifndef STAN_MATH_PRIM_SCAL_PROB_NORMAL_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_NORMAL_RNG_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline double
    normal_rng(const double mu,
               const double sigma,
               RNG& rng) {
      using boost::variate_generator;
      using boost::normal_distribution;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;

      static const char* function("stan::math::normal_rng");

      check_finite(function, "Location parameter", mu);
      check_not_nan(function, "Location parameter", mu);
      check_positive(function, "Scale parameter", sigma);
      check_not_nan(function, "Scale parameter", sigma);

      variate_generator<RNG&, normal_distribution<> >
        norm_rng(rng, normal_distribution<>(mu, sigma));
      return norm_rng();
    }
  }
}
#endif
