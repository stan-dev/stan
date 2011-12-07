#ifndef __STAN__PROB__DISTRIBUTIONS_LKJ_COV_HPP__
#define __STAN__PROB__DISTRIBUTIONS_LKJ_COV_HPP__

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/policies/policy.hpp>

#include "stan/prob/transform.hpp"
#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"
#include "stan/prob/distributions_lognormal.hpp"
#include "stan/prob/distributions_lkj_corr.hpp"

namespace stan {
  namespace prob {
    using namespace std;
    using namespace stan::maths;
    
    // ?? write these in terms of cpcs rather than corr matrix

    // LKJ_cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu vector, sigma > 0 vector, eta > 0 ]
    template <typename T_y, typename T_loc, typename T_scale, typename T_shape, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		const Matrix<T_loc,Dynamic,1>& mu,
		const Matrix<T_scale,Dynamic,1>& sigma,
		const T_shape& eta,
		const Policy& /* pol */) {
      const unsigned int K = y.rows();
      const Array<T_y,Dynamic,1> sds = y.diagonal().array().sqrt();
      T_shape log_prob = 0.0;
      for(unsigned int k = 0; k < K; k++) {
	log_prob += lognormal_log(log(sds(k,1)), mu(k,1), sigma(k,1));
      }
      if(eta == 1.0) {
	// no need to rescale y into a correlation matrix
	log_prob += lkj_corr_log(y,eta); 
	return log_prob;
      }
      DiagonalMatrix<double,Dynamic> D(K);
      D.diagonal() = sds.inverse();
      log_prob += lkj_corr_log(D * y * D, eta);
      return log_prob;
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		const Matrix<T_loc,Dynamic,1>& mu,
		const Matrix<T_scale,Dynamic,1>& sigma,
		const T_shape& eta) {
      return lkj_cov_log (y, mu, sigma, eta, boost::math::policies::policy<> ());
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_propto_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		       const Matrix<T_loc,Dynamic,1>& mu,
		       const Matrix<T_scale,Dynamic,1>& sigma,
		       const T_shape& eta,
		       const Policy& /* pol */) {
      return lkj_cov_log (y, mu, sigma, eta, Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_propto_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		       const Matrix<T_loc,Dynamic,1>& mu,
		       const Matrix<T_scale,Dynamic,1>& sigma,
		       const T_shape& eta) {
      return lkj_cov_propto_log (y, mu, sigma, eta, boost::math::policies::policy<>());
    }


    // LKJ_Cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu scalar, sigma > 0 scalar, eta > 0 ]
    template <typename T_y, typename T_loc, typename T_scale, typename T_shape, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		const T_loc& mu, const T_scale& sigma, const T_shape& eta, const Policy& /* pol */) {
      const unsigned int K = y.rows();
      const Array<T_y,Dynamic,1> sds = y.diagonal().array().sqrt();
      T_shape log_prob = 0.0;
      for(unsigned int k = 0; k < K; k++) {
	log_prob += lognormal_log(sds(k,1), mu, sigma);
      }
      if(eta == 1.0) {
	log_prob += lkj_corr_log(y,eta); // no need to rescale y into a correlation matrix
	return log_prob;
      }
      DiagonalMatrix<double,Dynamic> D(K);
      D.diagonal() = sds.inverse();
      log_prob += lkj_corr_log(D * y * D, eta);
      return log_prob;
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		const T_loc& mu, const T_scale& sigma, const T_shape& eta) {
      lkj_cov_log (y, mu, sigma, eta, boost::math::policies::policy<> ());
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_propto_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		       const T_loc& mu, const T_scale& sigma, const T_shape& eta, const Policy& /* pol */) {
      return lkj_cov_log (y, mu, sigma, eta, Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_propto_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		       const T_loc& mu, const T_scale& sigma, const T_shape& eta) {
      return lkj_cov_propto_log (y, mu, sigma, eta, boost::math::policies::policy<>());
    }

  }
}
#endif
