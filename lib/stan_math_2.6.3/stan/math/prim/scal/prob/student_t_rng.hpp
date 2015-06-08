#ifndef STAN_MATH_PRIM_SCAL_PROB_STUDENT_T_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_STUDENT_T_RNG_HPP

#include <boost/random/student_t_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline double
    student_t_rng(const double nu,
                  const double mu,
                  const double sigma,
                  RNG& rng) {
      using boost::variate_generator;
      using boost::random::student_t_distribution;

      static const char* function("stan::math::student_t_rng");

      using stan::math::check_positive_finite;
      using stan::math::check_finite;

      check_positive_finite(function, "Degrees of freedom parameter", nu);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Scale parameter", sigma);

      variate_generator<RNG&, student_t_distribution<> >
        rng_unit_student_t(rng, student_t_distribution<>(nu));
      return mu + sigma * rng_unit_student_t();
    }
  }
}
#endif
