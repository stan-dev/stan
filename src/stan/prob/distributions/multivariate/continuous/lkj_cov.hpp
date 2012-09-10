#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_COV_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_COV_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/matrix.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>

#include <stan/prob/distributions/univariate/continuous/lognormal.hpp>
#include <stan/prob/distributions/multivariate/continuous/lkj_corr.hpp>

namespace stan {

  namespace prob {

    // LKJ_cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu vector, sigma > 0 vector, eta > 0 ]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,1>& sigma,
                const T_shape& eta,
                const Policy&) {
      static const char* function = "stan::prob::lkj_cov_log<%1%>(%1%)";
      
      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale,T_shape>::type lp(0.0);
      if (!check_size_match(function, mu.rows(), sigma.rows(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, y.cols(), y.rows(), &lp, Policy()))
      return lp;
      if (!check_size_match(function, mu.rows(), y.rows(), &lp, Policy()))
        return lp;
      if (!check_positive(function, eta, "eta", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter, mu", &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter, sigma", 
                        &lp, Policy()))
        return lp;    
      for (int m = 0; m < y.rows(); ++m)
        for (int n = 0; n < y.cols(); ++n)
          if (!check_finite(function, y(m,n), "Covariance matrix, y(m,n)", &lp, Policy()))
            return lp;
      
      const unsigned int K = y.rows();
      const Eigen::Array<T_y,Eigen::Dynamic,1> sds
        = y.diagonal().array().sqrt();
      for (unsigned int k = 0; k < K; k++) {
        lp += lognormal_log<propto>(sds(k), mu(k), sigma(k), Policy());
      }
      if (stan::is_constant<typename stan::scalar_type<T_shape> >::value
          && eta == 1.0) {
        // no need to rescale y into a correlation matrix
        lp += lkj_corr_log<propto,T_y,T_shape,Policy>(y, eta, Policy()); 
        return lp;
      }
      Eigen::DiagonalMatrix<T_y,Eigen::Dynamic> D(K);
      D.diagonal() = sds.inverse();  // FIXME:  D.diagonal() inefficient
      lp += lkj_corr_log<propto,T_y,T_shape,Policy>(D * y * D, eta, Policy());
      return lp;
    }

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,1>& sigma,
                const T_shape& eta) {
      return lkj_cov_log<propto>(y,mu,sigma,eta,stan::math::default_policy());
    }


    template <typename T_y, typename T_loc, typename T_scale, typename T_shape, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,1>& sigma,
                const T_shape& eta,
                const Policy&) {
      return lkj_cov_log<false>(y,mu,sigma,eta,Policy());
    }


    template <typename T_y, typename T_loc, typename T_scale, typename T_shape> 
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,1>& sigma,
                const T_shape& eta) {
      return lkj_cov_log<false>(y,mu,sigma,eta,stan::math::default_policy());
    }



    // LKJ_Cov(y|mu,sigma,eta) [ y covariance matrix (not correlation matrix)
    //                         mu scalar, sigma > 0 scalar, eta > 0 ]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const T_loc& mu, 
                const T_scale& sigma, 
                const T_shape& eta, 
                const Policy&) {
      static const char* function = "stan::prob::lkj_cov_log<%1%>(%1%)";

      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale,T_shape>::type lp(0.0);
      if (!check_positive(function, eta, "eta", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter, mu", &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter, sigma", 
                        &lp, Policy()))
        return lp;
      
      const unsigned int K = y.rows();
      const Eigen::Array<T_y,Eigen::Dynamic,1> sds
        = y.diagonal().array().sqrt();
      for (unsigned int k = 0; k < K; k++) {
        lp += lognormal_log<propto>(sds(k), mu, sigma, Policy());
      }
      if (stan::is_constant<typename stan::scalar_type<T_shape> >::value
          && eta == 1.0) {
        // no need to rescale y into a correlation matrix
        lp += lkj_corr_log<propto>(y,eta,Policy()); 
        return lp;
      }
      Eigen::DiagonalMatrix<T_y,Eigen::Dynamic> D(K);
      D.diagonal() = sds.inverse();
      lp += lkj_corr_log<propto,T_y,T_shape,Policy>(D * y * D, eta, Policy());
      return lp;
    }

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const T_loc& mu, 
                const T_scale& sigma, 
                const T_shape& eta) {
      return lkj_cov_log<propto>(y,mu,sigma,eta,stan::math::default_policy());
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const T_loc& mu, 
                const T_scale& sigma, 
                const T_shape& eta, 
                const Policy&) {
      return lkj_cov_log<false>(y,mu,sigma,eta,Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale,T_shape>::type
    lkj_cov_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const T_loc& mu, 
                const T_scale& sigma, 
                const T_shape& eta) {
      return lkj_cov_log<false>(y,mu,sigma,eta,stan::math::default_policy());
    }


  }
}
#endif
