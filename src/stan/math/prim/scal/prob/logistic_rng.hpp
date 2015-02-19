#ifndef STAN__MATH__PRIM__SCAL__PROB__LOGISTIC_RNG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__LOGISTIC_RNG_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/mix/core/partials_vari.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/log1p.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/meta/prob_traits.hpp>

namespace stan {
  namespace prob {

    template <class RNG>
    inline double
    logistic_rng(const double mu,
                 const double sigma,
                 RNG& rng) {
      using boost::variate_generator;
      using boost::random::exponential_distribution;

      static const char* function("stan::prob::logistic_rng");
      
      using stan::math::check_positive_finite;
      using stan::math::check_finite;

      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      variate_generator<RNG&, exponential_distribution<> >
        exp_rng(rng, exponential_distribution<>(1));
      return mu - sigma * std::log(exp_rng() / exp_rng());
    }
  }
}
#endif
