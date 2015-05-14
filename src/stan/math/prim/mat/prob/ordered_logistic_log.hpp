#ifndef STAN_MATH_PRIM_MAT_PROB_ORDERED_LOGISTIC_LOG_HPP
#define STAN_MATH_PRIM_MAT_PROB_ORDERED_LOGISTIC_LOG_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/log1m_exp.hpp>
#include <stan/math/prim/scal/fun/log1p_exp.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_less_or_equal.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/mat/prob/categorical_rng.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline T log_inv_logit_diff(const T& alpha, const T& beta) {
      using std::exp;
      using stan::math::log1m_exp;
      using stan::math::log1p_exp;
      return beta + log1m_exp(alpha - beta) - log1p_exp(alpha)
        - log1p_exp(beta);
    }

    // y in 0, ..., K-1;   c.size()==K-2, c increasing,  lambda finite
    /**
     * Returns the (natural) log probability of the specified integer
     * outcome given the continuous location and specified cutpoints
     * in an ordered logistic model.
     *
     * <p>Typically the continous location
     * will be the dot product of a vector of regression coefficients
     * and a vector of predictors for the outcome.
     *
     * @tparam propto True if calculating up to a proportion.
     * @tparam T_loc Location type.
     * @tparam T_cut Cut-point type.
     * @param y Outcome.
     * @param lambda Location.
     * @param c Positive increasing vector of cutpoints.
     * @return Log probability of outcome given location and
     * cutpoints.

     * @throw std::domain_error If the outcome is not between 1 and
     * the number of cutpoints plus 2; if the cutpoint vector is
     * empty; if the cutpoint vector contains a non-positive,
     * non-finite value; or if the cutpoint vector is not sorted in
     * ascending order.
     */
    template <bool propto, typename T_lambda, typename T_cut>
    typename boost::math::tools::promote_args<T_lambda, T_cut>::type
    ordered_logistic_log(int y, const T_lambda& lambda,
                         const Eigen::Matrix<T_cut, Eigen::Dynamic, 1>& c) {
      using std::exp;
      using std::log;
      using stan::math::inv_logit;
      using stan::math::log1m;
      using stan::math::log1p_exp;

      static const char* function("stan::math::ordered_logistic");

      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_less;
      using stan::math::check_less_or_equal;
      using stan::math::check_greater;
      using stan::math::check_bounded;

      int K = c.size() + 1;

      check_bounded(function, "Random variable", y, 1, K);
      check_finite(function, "Location parameter", lambda);
      check_greater(function, "Size of cut points parameter", c.size(), 0);
      for (int i = 1; i < c.size(); ++i)
        check_greater(function, "Cut points parameter", c(i), c(i - 1));

      check_finite(function, "Cut points parameter", c(c.size()-1));
      check_finite(function, "Cut points parameter", c(0));

      // log(1 - inv_logit(lambda))
      if (y == 1)
        return -log1p_exp(lambda - c(0));

      // log(inv_logit(lambda - c(K-3)));
      if (y == K) {
        return -log1p_exp(c(K-2) - lambda);
      }

      // if (2 < y < K) { ... }
      // log(inv_logit(lambda - c(y-2)) - inv_logit(lambda - c(y-1)))
      return log_inv_logit_diff(c(y-2) - lambda,
                                c(y-1) - lambda);
    }

    template <typename T_lambda, typename T_cut>
    typename boost::math::tools::promote_args<T_lambda, T_cut>::type
    ordered_logistic_log(int y, const T_lambda& lambda,
                         const Eigen::Matrix<T_cut, Eigen::Dynamic, 1>& c) {
      return ordered_logistic_log<false>(y, lambda, c);
    }
  }
}

#endif
