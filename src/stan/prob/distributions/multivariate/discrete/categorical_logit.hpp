#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_LOGIT_LOG_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_LOGIT_LOG_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix/log_softmax.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/matrix/log_sum_exp.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // CategoricalLog(n|theta)  [0 < n <= N, theta unconstrained], no checking
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_logit_log(int n, 
                          const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& beta) {
      static const char* function = "stan::prob::categorical_logit_log(%1%)";

      using stan::math::check_bounded;
      using stan::math::check_finite;
      using stan::math::log_sum_exp;

      double lp = 0.0;
      check_bounded(function, n, 1, beta.size(),
                    "categorical outcome out of support",
                    &lp);
      
      check_finite(function, beta, "log odds parameter", &lp);
      
      if (!include_summand<propto,T_prob>::value)
        return 0.0;
        
      // FIXME:  wasteful vs. creating term (n-1) if not vectorized
      return beta(n-1) - log_sum_exp(beta); // == log_softmax(beta)(n-1);
    }

    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_logit_log(int n, 
                          const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& beta) {
      return categorical_logit_log<false>(n,beta);
    }

    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_logit_log(const std::vector<int>& ns, 
                          const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& beta) {
      static const char* function = "stan::prob::categorical_logit_log(%1%)";

      using stan::math::check_bounded;
      using stan::math::check_finite;
      using stan::math::log_softmax;
      using stan::math::sum;

      double lp = 0.0;
      for (size_t k = 0; k < ns.size(); ++k)
        check_bounded(function, ns[k], 1, beta.size(),
                      "categorical outcome out of support",
                      &lp);
      
      check_finite(function, beta, "log odds parameter", &lp);

      if (!include_summand<propto,T_prob>::value)
        return 0.0;
      
      if (ns.size() == 0)
        return 0.0;
        
      Eigen::Matrix<T_prob,Eigen::Dynamic,1> log_softmax_beta
        = log_softmax(beta);

      // FIXME:  replace with more efficient sum()
      Eigen::Matrix<typename boost::math::tools::promote_args<T_prob>::type,
                    Eigen::Dynamic,1> results(ns.size());
      for (size_t i = 0; i < ns.size(); ++i)
        results[i] = log_softmax_beta(ns[i] - 1);
      return sum(results);
    }

    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_logit_log(const std::vector<int>& ns,
                          const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& beta) {
      return categorical_logit_log<false>(ns,beta);
    }


  }
}
#endif
