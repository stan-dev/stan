#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_GP_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_GP_HPP__

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/meta/traits.hpp>
#include <stan/agrad/matrix.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/rows_dot_product.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/sum.hpp>

#include <stan/math/matrix/ldlt.hpp>

namespace stan {
  namespace prob {
    // MultiGP(y|Sigma,w)   [y.rows() = w.size(), y.cols() = Sigma.rows();
    //                            Sigma symmetric, non-negative, definite]
    /**
     * The log of a multivariate Gaussian Process for the given y, Sigma, and
     * w.  y is a dxN matrix, where each column is a different observation and each
     * row is a different output dimension.  The Guassian Process is assumed to
     * have a scaled kernel matrix with a different scale for each output dimension.
     * This distribution is equivalent to:
     *
     * @param y A pxn matrix of observations.
     * @param Z A pxm matrix. The design matrix.
     * @param sigma A p vector of observation covariance matrix.
     * @param T A mxm matrix. The transiation matrix.
     * @param RQR A mxm matrix. The state covariance matrix.
     * @return The log of the joint density of the DLM.
     * @throw std::domain_error if Sigma is not square, not symmetric, 
     * or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_covar Type of kernel.
     * @tparam T_w Type of weight.
     */
    template <bool propto,
              typename T_y, typename T_Z, typename T_T,
              typename T_sigma, typename T_RQR
              >
    typename boost::math::tools::promote_args<T_y,T_w,T_covar>::type
    dlm_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
            const Eigen::Matrix<T_Z,Eigen::Dynamic,Eigen::Dynamic>& Z,
            const Eigen::Matrix<T_T,Eigen::Dynamic,Eigen::Dynamic>& T,
            const Eigen::Matrix<T_sigma,Eigen::Dynamic,1>& sigma,
            const Eigen::Matrix<T_RQR,Eigen::Dynamic,Eigen::Dynamic>& RQR) {
      static const char* function = "stan::prob::dlm_log(%1%)";
      // 
      typedef typename boost::math::tools::promote_args<T_y,T_Z,T_T,T_sigma,T_RQR>::type T_lp;
      T_lp lp(0.0);
      
      using stan::math::log;
      using stan::math::sum;
      using stan::math::check_not_nan;
      using stan::math::check_size_match;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_symmetric;
      using stan::math::dot_product;
      using stan::math::rows_dot_product;
      using stan::math::log_determinant_ldlt;
      using stan::math::mdivide_right_ldlt;
      using stan::math::LDLT_factor;

      int p y.rows(); // number of variables
      int n y.rows(); // number of observations
      int m T.rows(); // number of states

      if (!check_size_match(function,
                            Z.rows(), "rows of Z",
                            y.cols(), "rows of y",
                            &lp))
        return lp;
      if (!check_size_match(function,
                            Z.cols(), "columns of Z",
                            T.rows(), "rows of T",
                            &lp))
        return lp;
      if (!check_size_match(function,
                            sigma.rows(), "length of sigma",
                            y.rows(), "rows of y",
                            &lp))
        return lp;
      if (!check_size_match(function,
                            T.rows(), "columns of T",
                            T.cols(), "rows of T",
                            &lp))
        return lp;
      if (!check_size_match(function,
                            RQR.rows(), "columns of RQR",
                            T.rows(), "rows of T",
                            &lp))
        return lp;
      if (!check_size_match(function,
                            RQR.rows(), "columns of RQR",
                            RQR.cols(), "rows of RQR",
                            &lp))
        return lp;

      // TODO: what check finite? 
      if (!check_symmetric(function, RQR, "RQR", &lp))
        return lp;

      if (y.cols() == 0 || y.rows == 0)
        return lp;


      if (include_summand<propto>::value) {
        lp += NEG_LOG_SQRT_TWO_PI * y.rows() * y.cols();
      }
      
      if (include_summand<propto,T_y,T_w,T_covar>::value) {
        // TODO: make arguments
        Eigen::Matrix<T_T,Eigen::Dynamic,1> a1 = Eigen::Zero(m, 1);
        Eigen::Matrix<T_T,Eigen::Dynamic,1> P1 = Eigen::Diagonal(m, m) * 10e6;

        for (int i = 0; i < n; ++i) {
          lp += 1;
        }
      }

      return lp;
    }
    
    // template <typename T_y, typename T_loc, typename T_covar>
    // inline
    // typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    // multi_gp_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
    //              const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
    //              const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& w) {
    //   return multi_gp_log<false>(y,Sigma,w);
    // }
  }    
}

#endif
