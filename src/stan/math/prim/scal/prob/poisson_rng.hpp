#ifndef STAN__MATH__PRIM__SCAL__PROB__POISSON_RNG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__POISSON_RNG_HPP

#include <limits>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/mix/core/partials_vari.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/meta/prob_traits.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>

namespace stan {

  namespace prob {

    static const double POISSON_MAX_RATE = pow(2,30);

    template <class RNG>
    inline int
    poisson_rng(const double lambda,
                RNG& rng) {
      using boost::variate_generator;
      using boost::random::poisson_distribution;

      static const char* function("stan::prob::poisson_rng");
      
      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::check_less;
 
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
