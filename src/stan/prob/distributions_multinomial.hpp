#ifndef __STAN__PROB__DISTRIBUTIONS_MULTINOMIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_MULTINOMIAL_HPP__

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

    using Eigen::Matrix;
    using Eigen::Dynamic;

    // Multinomial(ns|N,theta)   [0 <= n <= N;  SUM ns = N;   
    //                            0 <= theta[n] <= 1;  SUM theta = 1]
    template <typename T_prob, class Policy>
    inline typename boost::math::tools::promote_args<T_prob>::type
    multinomial_log(const std::vector<int>& ns,
		    const Matrix<T_prob,Dynamic,1>& theta, 
		    const Policy& /* pol */) {
      unsigned int len = ns.size();
      double sum = 1.0;
      for (unsigned int i = 0; i < len; ++i) 
	sum += ns[i];
      typename boost::math::tools::promote_args<T_prob>::type log_p
	= lgamma(sum);
      for (unsigned int i = 0; i < len; ++i)
	log_p -= lgamma(ns[i] + 1.0);
      for (unsigned int i = 0; i < len; ++i)
	log_p += ns[i] * log(theta[i]);
      return log_p;
    }

    template <typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    multinomial_log(const std::vector<int>& ns,
		    const Matrix<T_prob,Dynamic,1>& theta) {
      return multinomial_log (ns, theta, boost::math::policies::policy<>());
    }
    
    template <typename T_prob, class Policy>
    inline typename boost::math::tools::promote_args<T_prob>::type
    multinomial_propto_log(const std::vector<int>& ns,
			   const Matrix<T_prob,Dynamic,1>& theta, 
			   const Policy& /* pol */) {
      return multinomial_log (ns, theta, Policy());
    }

    template <typename T_prob>
    inline typename boost::math::tools::promote_args<T_prob>::type
    multinomial_propto_log(const std::vector<int>& ns,
			   const Matrix<T_prob,Dynamic,1>& theta) {
      return multinomial_propto_log (ns, theta, boost::math::policies::policy<>());
    }

  }
}
#endif
