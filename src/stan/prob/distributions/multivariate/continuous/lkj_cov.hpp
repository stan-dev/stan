#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_COV_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_COV_HPP

#include <stan/prob/constants.hpp>
#include <stan/math/matrix.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>

#include <stan/prob/distributions/univariate/continuous/lognormal.hpp>
#include <stan/prob/distributions/multivariate/continuous/lkj_corr.hpp>

namespace stan {

  namespace prob {

    // LKJ_cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu vector, sigma > 0 vector, eta > 0 ]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,1>& sigma,
                const T_shape& eta) {
      static const char* function = "stan::prob::lkj_cov_log(%1%)";
      
      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale,T_shape>::type lp(0.0);
      check_size_match(function, 
                       mu.rows(), "Rows of location parameter",
                       sigma.rows(), "columns of scale parameter",
                       &lp);
      check_size_match(function, 
                       y.rows(), "Rows of random variable",
                       y.cols(), "columns of random variable",
                       &lp);
      check_size_match(function, 
                       y.rows(), "Rows of random variable",
                       mu.rows(), "rows of location parameter",
                       &lp);
      check_positive(function, eta, "Shape parameter", &lp);
      check_finite(function, mu, "Location parameter", &lp);
      check_finite(function, sigma, "Scale parameter", &lp);
      // FIXME: build vectorized versions
      for (int m = 0; m < y.rows(); ++m)
        for (int n = 0; n < y.cols(); ++n)
          check_finite(function, y(m,n), "Covariance matrix", &lp);
      
      const unsigned int K = y.rows();
      const Eigen::Array<T_y,Eigen::Dynamic,1> sds
        = y.diagonal().array().sqrt();
      for (unsigned int k = 0; k < K; k++) {
        lp += lognormal_log<propto>(sds(k), mu(k), sigma(k));
      }
      if (stan::is_constant<typename stan::scalar_type<T_shape> >::value
          && eta == 1.0) {
        // no need to rescale y into a correlation matrix
        lp += lkj_corr_log<propto,T_y,T_shape>(y, eta); 
        return lp;
      }
      Eigen::DiagonalMatrix<T_y,Eigen::Dynamic> D(K);
      D.diagonal() = sds.inverse();
      lp += lkj_corr_log<propto,T_y,T_shape>(D * y * D, eta);
      return lp;
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape> 
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,1>& sigma,
                const T_shape& eta) {
      return lkj_cov_log<false>(y,mu,sigma,eta);
    }

    // LKJ_Cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu scalar, sigma > 0 scalar, eta > 0 ]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const T_loc& mu, 
                const T_scale& sigma, 
                const T_shape& eta) {
      static const char* function = "stan::prob::lkj_cov_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale,T_shape>::type lp(0.0);
      check_positive(function, eta, "Shape parameter", &lp);
      check_finite(function, mu, "Location parameter", &lp);
      check_finite(function, sigma, "Scale parameter", &lp);
      
      const unsigned int K = y.rows();
      const Eigen::Array<T_y,Eigen::Dynamic,1> sds
        = y.diagonal().array().sqrt();
      for (unsigned int k = 0; k < K; k++) {
        lp += lognormal_log<propto>(sds(k), mu, sigma);
      }
      if (stan::is_constant<typename stan::scalar_type<T_shape> >::value
          && eta == 1.0) {
        // no need to rescale y into a correlation matrix
        lp += lkj_corr_log<propto>(y,eta); 
        return lp;
      }
      Eigen::DiagonalMatrix<T_y,Eigen::Dynamic> D(K);
      D.diagonal() = sds.inverse();
      lp += lkj_corr_log<propto,T_y,T_shape>(D * y * D, eta);
      return lp;
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const T_loc& mu, 
                const T_scale& sigma, 
                const T_shape& eta) {
      return lkj_cov_log<false>(y,mu,sigma,eta);
    }


  }
}
#endif
