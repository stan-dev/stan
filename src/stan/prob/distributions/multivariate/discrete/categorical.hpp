#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_HPP__

#include <stan/prob/traits.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Categorical(n|theta)  [0 < n <= N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto,
              typename T_prob, 
              class Policy>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(int n, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta, 
                    const Policy&) {
      static const char* function = "stan::prob::categorical_log(%1%)";

      using stan::math::check_bounded;
      using stan::math::check_simplex;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      typename Eigen::Matrix<T_prob,Eigen::Dynamic,1>::size_type lb = 1;

      double lp = 0.0;
      if (!check_bounded(function, n, lb, theta.size(),
                         "Number of categories",
                         &lp, Policy()))
        return lp;
      
      if (!stan::is_constant_struct<T_prob>::value) {
        Eigen::Matrix<double,Eigen::Dynamic,1> theta_dbl(theta.size());
        for (int i = 0; i < theta_dbl.size(); ++i)
          theta_dbl(i) = value_of(theta(i));
        if (!check_simplex(function, theta,
                           "Probabilities parameter",
                           &lp, Policy()))
          return lp;
      } else {
        if (!check_simplex(function, theta,
                           "Probabilities parameter",
                           &lp, Policy()))
          return lp;
      }

      if (include_summand<propto,T_prob>::value)
        return log(theta(n-1));
      return 0.0;
    }

    template <bool propto,
              typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const typename Eigen::Matrix<T_prob,Eigen::Dynamic,1>::size_type n, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      return categorical_log<propto>(n,theta,stan::math::default_policy());
    }


    template <typename T_prob, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const typename Eigen::Matrix<T_prob,Eigen::Dynamic,1>::size_type n, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta, 
                    const Policy&) {
      return categorical_log<false>(n,theta,Policy());
    }

    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const typename Eigen::Matrix<T_prob,Eigen::Dynamic,1>::size_type n, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      return categorical_log<false>(n,theta,stan::math::default_policy());
    }


  }
}
#endif
