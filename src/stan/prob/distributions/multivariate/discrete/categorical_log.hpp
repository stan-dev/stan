#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_LOG_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_LOG_HPP__

#include <boost/math/tools/promotion.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // CategoricalLog(n|theta)  [0 < n <= N, theta unconstrained], no checking
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log_log(int n, 
                        const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& log_theta) {
      static const char* function = "stan::prob::categorical_log_log(%1%)";

      using stan::math::check_bounded;
      using stan::math::check_finite;

      double lp = 0.0;
      if (!check_bounded(function, n, 1, log_theta.size(),
                         "categorical outcome out of support",
                         &lp))
        return lp;

      if (!check_finite(function, log_theta, "log probability parameter", &lp))
        return lp;

      if (!include_summand<propto,T_prob>::value)
        return 0.0;
        
      return log_theta(n-1);
    }

    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log_log(int n,
                        const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& log_theta) {
      return categorical_log_log<false>(n,log_theta);
    }

    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log_log(const std::vector<int>& ns,
                        const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& log_theta) {
      static const char* function = "stan::prob::categorical_log_log(%1%)";

      using stan::math::check_bounded;
      using stan::math::check_finite;
      using stan::math::sum;

      double lp = 0.0;
      for (int i = 0; i < ns.size(); ++i)
        if (!check_bounded(function, ns[i], 1, log_theta.size(),
                           "element of categorical outcomes out of support",
                           &lp))
        return lp;

      if (!check_finite(function, log_theta, "log probability parameter", &lp))
        return lp;

      if (!include_summand<propto,T_prob>::value)
        return 0.0;

      if (ns.size() == 0)
        return 0.0;
      
      Eigen::Matrix<typename boost::math::tools::promote_args<T_prob>::type,
                    Eigen::Dynamic,1> log_theta_ns(ns.size());
    
      for (int i = 0; i < ns.size(); ++i)
        log_theta_ns(i) = log_theta(ns[i] - 1);
        
      return sum(log_theta_ns);
    }

    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log_log(const std::vector<int>& ns,
                        const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& log_theta) {
      return categorical_log_log<false>(ns,log_theta);
    }


  }
}
#endif
