#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/math/error_handling.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    // Categorical(n|theta)  [0 < n <= N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(int n, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      static const char* function = "stan::prob::categorical_log(%1%)";

      using stan::math::check_bounded;
      using stan::math::check_simplex;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      int lb = 1;

      double lp = 0.0;
      check_bounded(function, n, lb, theta.size(),
                    "Number of categories",
                    &lp);
      
      if (!stan::is_constant_struct<T_prob>::value) {
        Eigen::Matrix<double,Eigen::Dynamic,1> theta_dbl(theta.size());
        for (int i = 0; i < theta_dbl.size(); ++i)
          theta_dbl(i) = value_of(theta(i));
        if (!check_simplex(function, theta,
                           "Probabilities parameter",
                           &lp))
          return lp;
      } else {
        if (!check_simplex(function, theta,
                           "Probabilities parameter",
                           &lp))
          return lp;
      }

      if (include_summand<propto,T_prob>::value)
        return log(theta(n-1));
      return 0.0;
    }

    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const typename Eigen::Matrix<T_prob,Eigen::Dynamic,1>::size_type n, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      return categorical_log<false>(n,theta);
    }


    // Categorical(n|theta)  [0 < n <= N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const std::vector<int>& ns, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      static const char* function = "stan::prob::categorical_log(%1%)";

      using boost::math::tools::promote_args;
      using stan::math::check_bounded;
      using stan::math::check_simplex;
      using stan::math::sum;
      using stan::math::value_of;

      int lb = 1;

      double lp = 0.0;
      for (size_t i = 0; i < ns.size(); ++i)
        check_bounded(function, ns[i], lb, theta.size(),
                      "element of outcome array",
                      &lp);
      
      if (!stan::is_constant_struct<T_prob>::value) {
        Eigen::Matrix<double,Eigen::Dynamic,1> theta_dbl(theta.size());
        for (int i = 0; i < theta_dbl.size(); ++i)
          theta_dbl(i) = value_of(theta(i));
        if (!check_simplex(function, theta,
                           "Probabilities parameter",
                           &lp))
          return lp;
      } else {
        if (!check_simplex(function, theta,
                           "Probabilities parameter",
                           &lp))
          return lp;
      }

      if (!include_summand<propto,T_prob>::value)
        return 0.0;
      
      if (ns.size() == 0)
        return 0.0;

      Eigen::Matrix<T_prob,Eigen::Dynamic,1> log_theta(theta.size());
      for (int i = 0; i < theta.size(); ++i)
        log_theta(i) = log(theta(i));
      
      Eigen::Matrix<typename boost::math::tools::promote_args<T_prob>::type,
                    Eigen::Dynamic,1> log_theta_ns(ns.size());
      for (size_t i = 0; i < ns.size(); ++i)
        log_theta_ns(i) = log_theta(ns[i] - 1);
    
      return sum(log_theta_ns);
    }


    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const std::vector<int>& ns, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      return categorical_log<false>(ns,theta);
    }

    template <class RNG>
    inline int
    categorical_rng(const Eigen::Matrix<double,Eigen::Dynamic,1>& theta,
                    RNG& rng) {
      using boost::variate_generator;
      using boost::uniform_01;
      using stan::math::check_simplex;

      static const char* function = "stan::prob::categorical_rng(%1%)";

      check_simplex(function, theta,
                    "Probabilities parameter", (double*)0);

      variate_generator<RNG&, uniform_01<> >
        uniform01_rng(rng, uniform_01<>());
      
      Eigen::VectorXd index(theta.rows());
      index.setZero();

      for(int i = 0; i < theta.rows(); i++) {
        for(int j = i; j < theta.rows(); j++)
          index(j) += theta(i,0);
      }

      double c = uniform01_rng();
      int b = 0;
      while(c > index(b,0))
        b++;
      return b + 1;
    }
  }
}
#endif
