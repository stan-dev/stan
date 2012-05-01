#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__ORDERED_LOGISTIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__ORDERED_LOGISTIC_HPP__

#include <stan/prob/traits.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/constants.hpp>


namespace stan {

  namespace prob {


    template <typename T>
    inline T log_inv_logit_diff(const T& alpha, const T& beta) {
      using std::exp;
      using stan::math::log1m;
      using stan::math::log1p_exp;
      return beta + log1m(exp(alpha - beta)) - log1p_exp(alpha) - log1p_exp(beta);
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
     * @tparam Policy 

     * @param y Outcome.
     * @param lambda Location.
     * @param c Positive increasing vector of cutpoints.
     * @param Policy Error policy (only its type matters).
     * @return Log probability of outcome given location and
     * cutpoints.
     
     * @throw std::domain_error If the outcome is not between 1 and
     * the number of cutpoints plus 2; if the cutpoint vector is
     * empty; if the cutpoint vector contains a non-positive,
     * non-finite value; or if the cutpoint vector is not sorted in
     * ascending order.
     */
    template <bool propto,
              typename T_lambda,
              typename T_cut,
              class Policy>
    typename boost::math::tools::promote_args<T_lambda,T_cut>::type
    ordered_logistic_log(int y,  
                         const T_lambda& lambda,  
                         const Eigen::Matrix<T_cut,Eigen::Dynamic,1>& c, 
                         const Policy&) {

      using std::exp;
      using std::log;
      using stan::math::inv_logit;
      using stan::math::log1m;
      using stan::math::log1p_exp;

      static const char* function = "stan::prob::ordered_logistic<%1%>(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_nonnegative;
      using stan::math::check_less;
      using stan::math::check_less_or_equal;
      using stan::math::check_greater;
      using stan::math::check_bounded;

      int K = c.size() + 2;

      typename boost::math::tools::promote_args<T_lambda,T_cut>::type lp(0.0);
      if (!check_bounded(function, y, 1, K,
                         "y must be between 1 and K", 
                         &lp, Policy()))
        return lp;

      if (!check_finite(function, lambda, 
                        "location must be finite", &lp, Policy()))
        return lp;

      if (!check_greater(function, c.size(), 0,
                         "must be at least one cutpoint",
                         &lp, Policy()))
        return lp;

      if (!check_positive(function, c(0),
                           "first cut point must be positive",
                           &lp, Policy()))
        return lp;

      for (int i = 1; i < c.size(); ++i)
        if (!check_greater(function, c(i), c(i - 1),
                           "cut points must be positie increasing",
                           &lp, Policy()))
          return lp;

      if (!check_finite(function, c(c.size()-1), 
                        "the last cut point must be finite",
                        &lp, Policy()))
        return lp;
      
      if (!check_finite(function, c(0),
                        "the first cut point must be finite",
                        &lp, Policy())) 
        return lp;

      // log(1 - inv_logit(lambda))
      if (y == 1)
        return -log1p_exp(lambda); 

      // log(inv_logit(lambda) - inv_logit(lambda - c(0)))
      if (y == 2)
        return log_inv_logit_diff(-lambda, c(0) - lambda);

      // log(inv_logit(lambda - c(K-3)));
      if (y == K) {
        return -log1p_exp(c(K-3) - lambda);
      }

      // if (2 < y < K) { ... }
      // log(inv_logit(lambda - c(y-2)) - inv_logit(lambda - c(y-1)))
      return log_inv_logit_diff(c(y-3) - lambda, 
                                c(y-2) - lambda);

    }


    template <bool propto,
              typename T_lambda,
              typename T_cut>
    typename boost::math::tools::promote_args<T_lambda,T_cut>::type
    ordered_logistic_log(int y,  
                         const T_lambda& lambda,  
                         const Eigen::Matrix<T_cut,Eigen::Dynamic,1>& c) {
      return ordered_logistic_log<propto>(y,lambda,c,stan::math::default_policy());
    }


    template <typename T_lambda,
              typename T_cut,
              class Policy>
    typename boost::math::tools::promote_args<T_lambda,T_cut>::type
    ordered_logistic_log(int y,  
                         const T_lambda& lambda,  
                         const Eigen::Matrix<T_cut,Eigen::Dynamic,1>& c,
                         const Policy&) {
      return ordered_logistic_log<false>(y,lambda,c,Policy());
    }


    template <typename T_lambda,
              typename T_cut>
    typename boost::math::tools::promote_args<T_lambda,T_cut>::type
    ordered_logistic_log(int y,  
                         const T_lambda& lambda,  
                         const Eigen::Matrix<T_cut,Eigen::Dynamic,1>& c) {
      return ordered_logistic_log<false>(y,lambda,c,stan::math::default_policy());
    }

  }
}

#endif
