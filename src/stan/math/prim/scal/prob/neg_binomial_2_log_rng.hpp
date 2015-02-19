#ifndef STAN__MATH__PRIM__SCAL__PROB__NEG_BINOMIAL_2_LOG_RNG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__NEG_BINOMIAL_2_LOG_RNG_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/random/negative_binomial_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/mix/core/partials_vari.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/log_sum_exp.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/gamma_rng.hpp>
#include <stan/math/prim/scal/prob/poisson_rng.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>

namespace stan {

  namespace prob {

    template <class RNG>
    inline int
    neg_binomial_2_log_rng(const double eta,
                           const double phi,
                           RNG& rng) {
      using boost::variate_generator;
      using boost::random::negative_binomial_distribution;

      static const char* function("stan::prob::neg_binomial_2_log_rng");

      using stan::math::check_finite;
      using stan::math::check_positive_finite;

      check_finite(function, "Log-location parameter", eta);
      check_positive_finite(function, "Precision parameter", phi);


      return stan::prob::poisson_rng(stan::prob::gamma_rng(phi,
                                                           phi/std::exp(eta),
                                                           rng),
                                     rng);
    }
  }
}
#endif
