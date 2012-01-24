#ifndef __STAN__PROB__DISTRIBUTIONS__CATEGORICAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__CATEGORICAL_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    using Eigen::Dynamic;
    using Eigen::Matrix;

    // Categorical(n|theta)  [0 <= n < N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto = false, 
              typename T_prob, 
              class Policy = policy<> >
    inline typename promote_args<T_prob>::type
      categorical_log(const unsigned int n, const Matrix<T_prob,Dynamic,1>& theta, const Policy& = Policy()) {
      static const char* function = "stan::prob::categorical_log<%1%>(%1%)";

      typename promote_args<T_prob>::type lp(0.0);
      if (!check_bounded(function, n, 0U, theta.size()-1,
                         "Number of items, n,",
                         &lp, Policy()))
        return lp;
      if (!check_simplex(function, theta,
                         "Simplex, theta,",
                         &lp, Policy()))
        return lp;
  
      if (include_summand<propto,T_prob>::value)
        return log(theta(n));
      return 0.0;
    }

  }
}
#endif
