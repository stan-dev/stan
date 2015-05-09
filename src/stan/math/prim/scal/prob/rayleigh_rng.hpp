#ifndef STAN_MATH_PRIM_SCAL_PROB_RAYLEIGH_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_RAYLEIGH_RNG_HPP

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline double
    rayleigh_rng(const double sigma,
                 RNG& rng) {
      using boost::variate_generator;
      using boost::random::uniform_real_distribution;

      static const char* function("stan::math::rayleigh_rng");

      using stan::math::check_positive;

      check_positive(function, "Scale parameter", sigma);

      variate_generator<RNG&, uniform_real_distribution<> >
        uniform_rng(rng, uniform_real_distribution<>(0.0, 1.0));
      return sigma * std::sqrt(-2.0 * std::log(uniform_rng()));
    }
  }
}
#endif
