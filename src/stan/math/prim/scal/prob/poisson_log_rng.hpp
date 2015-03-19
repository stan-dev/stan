#ifndef STAN__MATH__PRIM__SCAL__PROB__POISSON_LOG_RNG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__POISSON_LOG_RNG_HPP

#include <limits>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/gamma_q.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>

namespace stan {

  namespace prob {

    static const double POISSON_MAX_LOG_RATE = 30 * log(2);

    template <class RNG>
    inline int
    poisson_log_rng(const double alpha,
                    RNG& rng) {
      using boost::variate_generator;
      using boost::random::poisson_distribution;

      static const char* function("stan::prob::poisson_log_rng");

      using stan::math::check_not_nan;
      using stan::math::check_nonnegative;
      using stan::math::check_less;
      using std::exp;

      check_not_nan(function, "Log rate parameter", alpha);
      check_less(function, "Log rate parameter", alpha, POISSON_MAX_LOG_RATE);

      variate_generator<RNG&, poisson_distribution<> >
        poisson_rng(rng, poisson_distribution<>(exp(alpha)));
      return poisson_rng();
    }
  }
}
#endif
