#ifndef __STAN__PROB__DISTRIBUTIONS_LKJ_CORR_HPP__
#define __STAN__PROB__DISTRIBUTIONS_LKJ_CORR_HPP__

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/policies/policy.hpp>

#include "stan/prob/transform.hpp"
#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"
#include "stan/prob/distributions_beta.hpp"

namespace stan {
  namespace prob {
    using namespace std;
    using namespace stan::maths;


    // ?? write these in terms of cpcs rather than corr matrix

    // LKJ_Corr(y|eta) [ y correlation matrix (not covariance matrix)
    //                  eta > 0 ]
    template <typename T_y, typename T_shape, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_shape>::type
    lkj_corr_log(const Matrix<T_y,Dynamic,Dynamic>& y, const T_shape& eta, const Policy& /* pol */) {
      // Lewandowski, Kurowicka, and Joe (2009) equations 15 and 16
      
      const unsigned int K = y.rows();
      T_shape the_sum = 0.0;
      T_shape constant = 0.0;
      T_shape beta_arg;
      
      if(eta == 1.0) {
	for(unsigned int k = 1; k < K; k++) { // yes, go from 1 to K - 1
	  beta_arg = 0.5 * (k + 1.0);
	  constant += k * beta_log(beta_arg, beta_arg);
	  the_sum += pow(static_cast<double>(k),2.0);
	}
	constant += the_sum * LOG_TWO;
	return constant;
      }

      T_shape diff;
      for(unsigned int k = 1; k < K; k++) { // yes, go from 1 to K - 1
	diff = K - k;
	beta_arg = eta + 0.5 * (diff - 1);
	constant += diff * beta_log(beta_arg, beta_arg);
	the_sum += (2.0 * eta - 2.0 + diff) * diff;
      }
      constant += the_sum * LOG_TWO;
      return (eta - 1.0) * log(y.determinant()) + constant;
    }
    
    template <typename T_y, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y, T_shape>::type
    lkj_corr_log(const Matrix<T_y,Dynamic,Dynamic>& y, const T_shape& eta) {
      return lkj_corr_log (y, eta, boost::math::policies::policy<>());
    }

    template <typename T_y, typename T_shape, class Policy>
    inline typename boost::math::tools::promote_args<T_y, T_shape>::type
    lkj_corr_propto_log(const Matrix<T_y,Dynamic,Dynamic>& y, const T_shape& eta, const Policy& /* pol */) {
      return lkj_corr_log (y, eta, Policy());
    }

    template <typename T_y, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y, T_shape>::type
    lkj_corr_propto_log(const Matrix<T_y,Dynamic,Dynamic>& y, const T_shape& eta) {
      return lkj_corr_propto_log (y, eta, boost::math::policies::policy<>());
    }

  }
}
#endif
