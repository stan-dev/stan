#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__CATEGORICAL_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/error_handling/matrix/check_simplex.hpp>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
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
      static const std::string function("stan::prob::categorical_log");

      using stan::error_handling::check_bounded;
      using stan::error_handling::check_simplex;
      using boost::math::tools::promote_args;
      using stan::math::value_of;

      int lb = 1;

      T_prob lp = 0.0;
      check_bounded(function, "Number of categories", n, lb, theta.size());
      
      if (!stan::is_constant_struct<T_prob>::value) {
        if (!check_simplex(function, "Probabilities parameter", theta))
          return lp;
      } else {
        if (!check_simplex(function, "Probabilities parameter", theta))
          return lp;
      }

      if (include_summand<propto,T_prob>::value)
        return log(theta(n-1));
      return 0.0;
    }

    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const typename 
                    math::index_type<Eigen::Matrix<T_prob,
                                                   Eigen::Dynamic,1> >::type n, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {

      return categorical_log<false>(n,theta);
    }


    // Categorical(n|theta)  [0 < n <= N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const std::vector<int>& ns, 
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      static const std::string function("stan::prob::categorical_log");

      using boost::math::tools::promote_args;
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_simplex;
      using stan::math::sum;
      using stan::math::value_of;

      int lb = 1;

      T_prob lp = 0.0;
      for (size_t i = 0; i < ns.size(); ++i)
        check_bounded(function, "element of outcome array", ns[i], lb, theta.size());
      
      if (!stan::is_constant_struct<T_prob>::value) {
        if (!check_simplex(function, "Probabilities parameter", theta))
          return lp;
      } else {
        if (!check_simplex(function, "Probabilities parameter", theta))
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
      using stan::error_handling::check_simplex;

      static const std::string function("stan::prob::categorical_rng");

      check_simplex(function, "Probabilities parameter", theta);

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
      while (c > index(b,0))
        b++;
      return b + 1;
    }
  }
}
#endif
