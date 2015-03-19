#ifndef STAN__MATH__PRIM__MAT__PROB__LKJ_COV_LOG_HPP
#define STAN__MATH__PRIM__MAT__PROB__LKJ_COV_LOG_HPP

#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>

#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/lognormal_log.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_log.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

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
      static const char* function("stan::prob::lkj_cov_log");

      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_square;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_scale,T_shape>::type lp(0.0);
      check_size_match(function,
                       "Rows of location parameter", mu.rows(),
                       "columns of scale parameter", sigma.rows());
      check_square(function, "random variable", y);
      check_size_match(function,
                       "Rows of random variable", y.rows(),
                       "rows of location parameter", mu.rows());
      check_positive(function, "Shape parameter", eta);
      check_finite(function, "Location parameter", mu);
      check_finite(function, "Scale parameter", sigma);
      // FIXME: build vectorized versions
      for (int m = 0; m < y.rows(); ++m)
        for (int n = 0; n < y.cols(); ++n)
          check_finite(function, "Covariance matrix", y(m,n));

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
      static const char* function("stan::prob::lkj_cov_log");

      using stan::math::check_finite;
      using stan::math::check_positive;
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_loc,T_scale,T_shape>::type lp(0.0);
      check_positive(function, "Shape parameter", eta);
      check_finite(function, "Location parameter", mu);
      check_finite(function, "Scale parameter", sigma);

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
