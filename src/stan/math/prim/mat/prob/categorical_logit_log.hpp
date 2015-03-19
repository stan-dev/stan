#ifndef STAN__MATH__PRIM__MAT__PROB__CATEGORICAL_LOGIT_LOG_HPP
#define STAN__MATH__PRIM__MAT__PROB__CATEGORICAL_LOGIT_LOG_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/arr/fun/log_sum_exp.hpp>
#include <stan/math/prim/mat/fun/log_softmax.hpp>
#include <stan/math/prim/mat/fun/log_sum_exp.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace prob {

    // CategoricalLog(n|theta)  [0 < n <= N, theta unconstrained], no checking
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    categorical_logit_log(int n,
                          const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& beta) {
      static const char* function("stan::prob::categorical_logit_log");

      using stan::math::check_bounded;
      using stan::math::check_finite;
      using stan::math::log_sum_exp;

      check_bounded(function, "categorical outcome out of support", n, 1, beta.size());
      check_finite(function, "log odds parameter", beta);

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
      static const char* function("stan::prob::categorical_logit_log");

      using stan::math::check_bounded;
      using stan::math::check_finite;
      using stan::math::log_softmax;
      using stan::math::sum;

      for (size_t k = 0; k < ns.size(); ++k)
        check_bounded(function, "categorical outcome out of support", ns[k], 1, beta.size());
      check_finite(function, "log odds parameter", beta);

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
