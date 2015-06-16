#ifndef STAN_MATH_PRIM_MAT_PROB_CATEGORICAL_LOG_HPP
#define STAN_MATH_PRIM_MAT_PROB_CATEGORICAL_LOG_HPP

#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/mat/err/check_simplex.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <cmath>
#include <vector>

namespace stan {

  namespace math {

    // Categorical(n|theta)  [0 < n <= N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(int n,
                    const Eigen::Matrix<T_prob, Eigen::Dynamic, 1>& theta) {
      static const char* function("stan::math::categorical_log");

      using stan::math::check_bounded;
      using stan::math::check_simplex;
      using boost::math::tools::promote_args;
      using stan::math::value_of;
      using std::log;

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

      if (include_summand<propto, T_prob>::value)
        return log(theta(n-1));
      return 0.0;
    }

    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const typename
                    math::index_type<Eigen::Matrix<T_prob,
                    Eigen::Dynamic, 1> >::type n,
                    const Eigen::Matrix<T_prob, Eigen::Dynamic, 1>& theta) {
      return categorical_log<false>(n, theta);
    }


    // Categorical(n|theta)  [0 < n <= N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const std::vector<int>& ns,
                    const Eigen::Matrix<T_prob, Eigen::Dynamic, 1>& theta) {
      static const char* function("stan::math::categorical_log");

      using boost::math::tools::promote_args;
      using stan::math::check_bounded;
      using stan::math::check_simplex;
      using stan::math::sum;
      using stan::math::value_of;
      using std::log;

      int lb = 1;

      T_prob lp = 0.0;
      for (size_t i = 0; i < ns.size(); ++i)
        check_bounded(function, "element of outcome array", ns[i],
                      lb, theta.size());

      if (!stan::is_constant_struct<T_prob>::value) {
        if (!check_simplex(function, "Probabilities parameter", theta))
          return lp;
      } else {
        if (!check_simplex(function, "Probabilities parameter", theta))
          return lp;
      }

      if (!include_summand<propto, T_prob>::value)
        return 0.0;

      if (ns.size() == 0)
        return 0.0;

      Eigen::Matrix<T_prob, Eigen::Dynamic, 1> log_theta(theta.size());
      for (int i = 0; i < theta.size(); ++i)
        log_theta(i) = log(theta(i));

      Eigen::Matrix<typename boost::math::tools::promote_args<T_prob>::type,
                    Eigen::Dynamic, 1> log_theta_ns(ns.size());
      for (size_t i = 0; i < ns.size(); ++i)
        log_theta_ns(i) = log_theta(ns[i] - 1);

      return sum(log_theta_ns);
    }


    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const std::vector<int>& ns,
                    const Eigen::Matrix<T_prob, Eigen::Dynamic, 1>& theta) {
      return categorical_log<false>(ns, theta);
    }

  }
}
#endif
