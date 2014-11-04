#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__ORDERED_LOGISTIC_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__ORDERED_LOGISTIC_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/error_handling.hpp>
#include <stan/math/functions/inv_logit.hpp>
#include <stan/math/functions/log1m.hpp>
#include <stan/math/functions/log1m_exp.hpp>
#include <stan/math/functions/log1p_exp.hpp>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_greater.hpp>
#include <stan/error_handling/scalar/check_less.hpp>
#include <stan/error_handling/scalar/check_less_or_equal.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/multivariate/discrete/categorical.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    template <typename T>
    inline T log_inv_logit_diff(const T& alpha, const T& beta) {
      using std::exp;
      using stan::math::log1m_exp;
      using stan::math::log1p_exp;
      return beta + log1m_exp(alpha - beta) - log1p_exp(alpha) - log1p_exp(beta);
    }
 
    // y in 0,...,K-1;   c.size()==K-2,  c increasing,  lambda finite
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
     * @tparam Policy Error policy (only its type matters). 

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
    typename boost::math::tools::promote_args<T_lambda,T_cut>::type
    ordered_logistic_log(int y, const T_lambda& lambda,  
                         const Eigen::Matrix<T_cut,Eigen::Dynamic,1>& c) {
      using std::exp;
      using std::log;
      using stan::math::inv_logit;
      using stan::math::log1m;
      using stan::math::log1p_exp;

      static const std::string function("stan::prob::ordered_logistic");
      
      using stan::error_handling::check_finite;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_less;
      using stan::error_handling::check_less_or_equal;
      using stan::error_handling::check_greater;
      using stan::error_handling::check_bounded;

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
    typename boost::math::tools::promote_args<T_lambda,T_cut>::type
    ordered_logistic_log(int y, const T_lambda& lambda,  
                         const Eigen::Matrix<T_cut,Eigen::Dynamic,1>& c) {
      return ordered_logistic_log<false>(y,lambda,c);
    }

    template <class RNG>
    inline int
    ordered_logistic_rng(const double eta,
                         const Eigen::Matrix<double,Eigen::Dynamic,1>& c,
                         RNG& rng) {
      using boost::variate_generator;
      using stan::math::inv_logit;

      static const std::string function("stan::prob::ordered_logistic");
      
      using stan::error_handling::check_finite;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_less;
      using stan::error_handling::check_less_or_equal;
      using stan::error_handling::check_greater;
      using stan::error_handling::check_bounded;

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
