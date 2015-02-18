#ifndef STAN__MATH__PRIM__SCAL__PROB__EXPONENTIAL_RNG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__EXPONENTIAL_RNG_HPP

#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/mix/core/partials_vari.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/meta/prob_traits.hpp>

namespace stan {
  
  namespace prob {

    template <class RNG>
    inline double
    exponential_rng(const double beta,
                    RNG& rng) {
      using boost::variate_generator;
      using boost::exponential_distribution;

      static const char* function("stan::prob::exponential_rng");

      using stan::math::check_positive_finite;

      check_positive_finite(function, "Inverse scale parameter", beta);

      variate_generator<RNG&, exponential_distribution<> >
        exp_rng(rng, exponential_distribution<>(beta));
      return exp_rng();
    }
  }
}

#endif
