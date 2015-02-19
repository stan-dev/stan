#ifndef STAN__MATH__PRIM__SCAL__PROB__NEG_BINOMIAL_RNG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__NEG_BINOMIAL_RNG_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/mix/core/partials_vari.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/gamma_rng.hpp>
#include <stan/math/prim/scal/prob/poisson_rng.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <stan/math/fwd/scal/fun/inc_beta.hpp>
#include <stan/math/rev/scal/fun/inc_beta.hpp>

namespace stan {

  namespace prob {

    template <class RNG>
    inline int
    neg_binomial_rng(const double alpha,
                     const double beta,
                     RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;

      static const char* function("stan::prob::neg_binomial_rng");

      using stan::math::check_positive_finite;      

      check_positive_finite(function, "Shape parameter", alpha);
      check_positive_finite(function, "Inverse scale parameter", beta);

      return stan::prob::poisson_rng(stan::prob::gamma_rng(alpha, beta,
                                                           rng),rng);
    }
  }
}
#endif
