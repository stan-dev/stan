#ifndef STAN__MATH__PRIM__SCAL__PROB__EXP_MOD_NORMAL_RNG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__EXP_MOD_NORMAL_RNG_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/mix/core/partials_vari.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/normal.hpp>
#include <stan/math/prim/scal/prob/exponential_rng.hpp>
#include <stan/math/prim/scal/meta/prob_traits.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>

namespace stan {

  namespace prob {

    template <class RNG>
    inline double
    exp_mod_normal_rng(const double mu,
                       const double sigma,
                       const double lambda,
                       RNG& rng) {

      static const char* function("stan::prob::exp_mod_normal_rng");

      using stan::math::check_positive_finite;
      using stan::math::check_finite;

      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_positive_finite(function, "Scale parameter", sigma);

      return stan::prob::normal_rng(mu, sigma,rng) 
        + stan::prob::exponential_rng(lambda, rng);
    }
  }
}
#endif



