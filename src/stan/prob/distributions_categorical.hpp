#ifndef __STAN__PROB__DISTRIBUTIONS_CATEGORICAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_CATEGORICAL_HPP__

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/policies/policy.hpp>

#include <Eigen/Dense>

#include "stan/prob/transform.hpp"
#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"



namespace stan {
  namespace prob {
    using namespace std;
    using namespace stan::maths;

    using Eigen::Dynamic;
    using Eigen::Matrix;

    // Categorical(n|theta)  [0 <= n < N;   0 <= theta[n] <= 1;  SUM theta = 1]
    template <typename T_prob, class Policy>
    inline typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const unsigned int n, const Matrix<T_prob,Dynamic,1>& theta, const Policy& /* pol */) {
      // FIXME: domain checks
      return log(theta(n));
    }

    template <typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    categorical_log(const unsigned int n, const Matrix<T_prob,Dynamic,1>& theta) {
      return categorical_log (n, theta, boost::math::policies::policy<>());
    }


    template <typename T_prob, class Policy>
    inline typename boost::math::tools::promote_args<T_prob>::type
    categorical_propto_log(const unsigned int n, const Matrix<T_prob,Dynamic,1>& theta, const Policy& /* pol */) {
      return categorical_log (n, theta, Policy());
    }

    template <typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    categorical_propto_log(const unsigned int n, const Matrix<T_prob,Dynamic,1>& theta) {
      return categorical_propto_log (n, theta, boost::math::policies::policy<>());
    }

  }
}
#endif
