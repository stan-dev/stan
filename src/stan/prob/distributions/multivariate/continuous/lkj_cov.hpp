#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_COV_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_COV_HPP__

#include <Eigen/Dense>

#include <stan/prob/constants.hpp>
#include <stan/maths/matrix_error_handling.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

#include <stan/prob/distributions/univariate/continuous/lognormal.hpp>
#include <stan/prob/distributions/multivariate/continuous/lkj_corr.hpp>

namespace stan {
  namespace prob {
    // LKJ_cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu vector, sigma > 0 vector, eta > 0 ]
    template <bool propto = false,
              typename T_y, typename T_loc, typename T_scale, typename T_shape, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const typename stan::maths::EigenType<T_y>::matrix& y,
                const typename stan::maths::EigenType<T_loc>::vector& mu,
                const typename stan::maths::EigenType<T_scale>::vector& sigma,
                const T_shape& eta,
                const Policy& = Policy()) {
      static const char* function = "stan::prob::lkj_cov_log<%1%>(%1%)";
      
      using stan::maths::check_size_match;
      using stan::maths::check_finite;
      using stan::maths::check_positive;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale,T_shape>::type lp(0.0);
      if (!check_size_match(function, mu.rows(), sigma.rows(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, mu.rows(), y.rows(), &lp, Policy()))
        return lp;
      if (!check_positive(function, eta, "eta", &lp, Policy()))
	return lp;
      if (!check_finite(function, mu, "Location parameter, mu", &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter, sigma", &lp, Policy()))
        return lp;      
      if (!check_finite(function, y, "Covariance matrix, y", &lp, Policy()))
        return lp;
      
      const unsigned int K = y.rows();
      const Eigen::Array<T_y,Eigen::Dynamic,1> sds = y.diagonal().array().sqrt();
      for(unsigned int k = 0; k < K; k++) {
        lp += lognormal_log<propto>(sds(k), mu(k), sigma(k));
      }
      if(eta == 1.0) {
        // no need to rescale y into a correlation matrix
        lp += lkj_corr_log<propto>(y,eta); 
        return lp;
      }
      Eigen::DiagonalMatrix<double,Eigen::Dynamic> D(K);
      D.diagonal() = sds.inverse();
      lp += lkj_corr_log<propto>(D * y * D, eta);
      return lp;
    }

    // LKJ_Cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu scalar, sigma > 0 scalar, eta > 0 ]
    template <bool propto = false,
              typename T_y, typename T_loc, typename T_scale, typename T_shape, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const T_loc& mu, 
                const T_scale& sigma, 
                const T_shape& eta, 
                const Policy& = Policy()) {
      static const char* function = "stan::prob::lkj_cov_log<%1%>(%1%)";

      using stan::maths::check_finite;
      using stan::maths::check_positive;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale,T_shape>::type lp(0.0);
      if (!check_positive(function, eta, "eta", &lp, Policy()))
	return lp;
      if (!check_finite(function, mu, "Location parameter, mu", &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter, sigma", &lp, Policy()))
        return lp;
      
      const unsigned int K = y.rows();
      const Eigen::Array<T_y,Eigen::Dynamic,1> sds = y.diagonal().array().sqrt();
      for(unsigned int k = 0; k < K; k++) {
        lp += lognormal_log<propto>(sds(k), mu, sigma);
      }
      if(eta == 1.0) {
        lp += lkj_corr_log<propto>(y,eta); // no need to rescale y into a correlation matrix
        return lp;
      }
      Eigen::DiagonalMatrix<double,Eigen::Dynamic> D(K);
      D.diagonal() = sds.inverse();
      lp += lkj_corr_log<propto>(D * y * D, eta);
      return lp;
    }
  }
}
#endif
