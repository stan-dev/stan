#ifndef STAN_MATH_PRIM_SCAL_PROB_GAMMA_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_GAMMA_RNG_HPP

#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/gamma_p.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_gamma.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline double
    gamma_rng(const double alpha,
              const double beta,
              RNG& rng) {
      using boost::variate_generator;
      using boost::gamma_distribution;

      static const char* function("stan::math::gamma_rng");

      using stan::math::check_positive_finite;

      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Inverse scale parameter", beta);

      /*
        the boost gamma distribution is defined by
        shape and scale, whereas the stan one is defined
        by shape and rate
      */
      variate_generator<RNG&, gamma_distribution<> >
        gamma_rng(rng, gamma_distribution<>(alpha, 1.0 / beta));
      return gamma_rng();
    }

  }
}

#endif
