#ifndef __STAN__PROB__DISTRIBUTIONS__LKJ_COV_HPP__
#define __STAN__PROB__DISTRIBUTIONS__LKJ_COV_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>

#include <stan/prob/distributions/lognormal.hpp>
#include <stan/prob/distributions/lkj_corr.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    using Eigen::Matrix;
    using Eigen::Dynamic;
    
    // ?? write these in terms of cpcs rather than corr matrix

    // LKJ_cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu vector, sigma > 0 vector, eta > 0 ]
    template <bool propto = false,
	      typename T_y, typename T_loc, typename T_scale, typename T_shape, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		const Matrix<T_loc,Dynamic,1>& mu,
		const Matrix<T_scale,Dynamic,1>& sigma,
		const T_shape& eta,
		const Policy& = Policy()) {
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


    // LKJ_Cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu scalar, sigma > 0 scalar, eta > 0 ]
    template <bool propto = false,
	      typename T_y, typename T_loc, typename T_scale, typename T_shape, 
	      class Policy = policy<> >
    inline typename promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Matrix<T_y,Dynamic,Dynamic>& y,
		const T_loc& mu, const T_scale& sigma, const T_shape& eta, const Policy& = Policy()) {
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


  }
}
#endif
