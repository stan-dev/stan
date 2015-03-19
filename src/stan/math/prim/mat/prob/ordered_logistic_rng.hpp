#ifndef STAN__MATH__PRIM__MAT__PROB__ORDERED_LOGISTIC_RNG_HPP
#define STAN__MATH__PRIM__MAT__PROB__ORDERED_LOGISTIC_RNG_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <stan/math/prim/scal/fun/log1p_exp.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/mat/prob/categorical_rng.hpp>

namespace stan {

  namespace prob {

    template <class RNG>
    inline int
    ordered_logistic_rng(const double eta,
                         const Eigen::Matrix<double,Eigen::Dynamic,1>& c,
                         RNG& rng) {
      using boost::variate_generator;
      using stan::math::inv_logit;

      static const char* function("stan::prob::ordered_logistic");

      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_less;
      using stan::math::check_less_or_equal;
      using stan::math::check_greater;
      using stan::math::check_bounded;

      check_finite(function, "Location parameter", eta);
      check_greater(function, "Size of cut points parameter", c.size(), 0);
      for (int i = 1; i < c.size(); ++i) {
        check_greater(function, "Cut points parameter", c(i), c(i - 1));
      }
      check_finite(function, "Cut points parameter", c(c.size()-1));
      check_finite(function, "Cut points parameter", c(0));

      Eigen::VectorXd cut(c.rows()+1);
      cut(0) = 1 - inv_logit(eta - c(0));
      for(int j = 1; j < c.rows(); j++)
        cut(j) = inv_logit(eta - c(j - 1)) - inv_logit(eta - c(j));
      cut(c.rows()) = inv_logit(eta - c(c.rows() - 1));

      return stan::prob::categorical_rng(cut, rng);
    }
  }
}

#endif
