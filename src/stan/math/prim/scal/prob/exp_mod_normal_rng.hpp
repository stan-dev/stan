#ifndef STAN_MATH_PRIM_SCAL_PROB_EXP_MOD_NORMAL_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_EXP_MOD_NORMAL_RNG_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/prob/normal_rng.hpp>
#include <stan/math/prim/scal/prob/exponential_rng.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline double
    exp_mod_normal_rng(const double mu,
                       const double sigma,
                       const double lambda,
                       RNG& rng) {
      static const char* function("stan::math::exp_mod_normal_rng");

      using stan::math::check_positive_finite;
      using stan::math::check_finite;

      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_positive_finite(function, "Scale parameter", sigma);

      return stan::math::normal_rng(mu, sigma, rng)
        + stan::math::exponential_rng(lambda, rng);
    }
  }
}
#endif



