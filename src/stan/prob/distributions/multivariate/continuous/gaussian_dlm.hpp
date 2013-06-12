#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__GAUSSIAN_DLM_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__GAUSSIAN_DLM_HPP__

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
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/add.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/math/matrix/inverse.hpp>
#include <stan/math/matrix/col.hpp>

// TODO: y as vector of vectors or matrix?

namespace stan {
  namespace prob {
    /**
     * The log of a multivariate Gaussian Process for the given y, Sigma, and
     * w.  y is a dxN matrix, where each column is a different observation and each
     * row is a different output dimension.  The Guassian Process is assumed to
     * have a scaled kernel matrix with a different scale for each output dimension.
     * This distribution is equivalent to, for \f$t = 1:N$,
     * \f{eqnarray*}{
     * y_t & \sim N(F' \theta_t, V) \\
     * \theta_t & \sim N(G \theta_{t-1}, W) \\
     * \theta_0 & \sim N(0, diag(10^{6}))
     * }
     *
     * @param y A r x T matrix of observations.
     * @param F A n x r matrix. The design matrix.
     * @param G A n x n matrix. The transition matrix.
     * @param V A r x r matrix. The observation covariance matrix.
     * @param W A n x n matrix. The state covariance matrix.
     * @return The log of the joint density of the GDLM.
     * @throw std::domain_error if Sigma is not square, not symmetric, 
     * or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_F Type of design matrix.
     * @tparam T_G Type of transition matrix.
     * @tparam T_V Type of observation covariance matrix.
     * @tparam T_W Type of state covariance matrix.
     */
    template <bool propto,
              typename T_y, 
              typename T_F, typename T_G,
              typename T_V, typename T_W
              >
    typename boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type
    gaussian_dlm_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_F,Eigen::Dynamic,Eigen::Dynamic>& F,
                     const Eigen::Matrix<T_G,Eigen::Dynamic,Eigen::Dynamic>& G,
                     const Eigen::Matrix<T_V,Eigen::Dynamic,Eigen::Dynamic>& V,
                     const Eigen::Matrix<T_W,Eigen::Dynamic,Eigen::Dynamic>& W) {
      static const char* function = "stan::prob::dlm_log(%1%)";
      typedef typename boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type T_lp;
      T_lp lp(0.0);
      
      using stan::math::check_not_nan;
      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_cov_matrix;
      using stan::math::add;
      using stan::math::multiply;
      using stan::math::transpose;
      using stan::math::inverse;
      using stan::math::subtract;

      int r = y.rows(); // number of variables
      int T = y.cols(); // number of observations
      int n = G.rows(); // number of states

      // check F
      if (!check_size_match(function,
                            F.cols(), "columns of F",
                            y.rows(), "rows of y",
                            &lp))
        return lp;
      if (!check_size_match(function,
                            F.rows(), "rows of F",
                            G.rows(), "rows of G",
                            &lp))
        return lp;
      // check G
      if (!check_size_match(function,
                            G.rows(), "rows of G",
                            G.cols(), "columns of G",
                            &lp))
        return lp;
      // check V
      if (!check_cov_matrix(function, V, "V", &lp))
        return lp;
      if (!check_size_match(function,
                            V.rows(), "rows of V",
                            y.rows(), "rows of y",
                            &lp))
        return lp;
      // check W
      if (!check_cov_matrix(function, W, "W", &lp))
        return lp;
      if (!check_size_match(function,
                            W.rows(), "rows of W",
                            G.rows(), "rows of G",
                            &lp))
        return lp;

      if (y.cols() == 0 || y.rows() == 0)
        return lp;

      if (include_summand<propto>::value) {
        lp += 0.5 *NEG_LOG_SQRT_TWO_PI * r * T;
      }
      
      if (include_summand<propto,T_y,T_F,T_G,T_V,T_W>::value) {
        Eigen::Matrix<typename 
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, 1> m(n);
        for (int i = 0; i < m.size(); i ++)
          m(i) = 0;
        Eigen::Matrix<typename 
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, Eigen::Dynamic> C(n, n);
        for (int i = 0; i < C.rows(); i ++) {
          for (int j = 0; j < C.cols(); j ++) {          
            if (i == j) {
              C(i, j) == 10e6;
            } else {
              C(i, j) == 0.0;
            }
          }
        }

        Eigen::Matrix<typename 
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, 1> a(n);
        Eigen::Matrix<typename
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, Eigen::Dynamic> R(n, n);
        Eigen::Matrix<typename
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, 1> f(r);
        Eigen::Matrix<typename
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, Eigen::Dynamic> Q(r, r);
        Eigen::Matrix<typename
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, Eigen::Dynamic> Q_inv(r, r);
        Eigen::Matrix<typename
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, 1> e(r);
        Eigen::Matrix<typename
                      boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type,
                      Eigen::Dynamic, Eigen::Dynamic> A(n, r);
        Eigen::Matrix<typename
                      boost::math::tools::promote_args<T_y>::type,
                      Eigen::Dynamic, 1> yi(r);

        for (int i = 0; i < T; ++i) {
          yi = y.col(i);
          std::cout << yi << std::endl;
          // Predict
          a = multiply(G, m);
          R = quad_form_sym(C, transpose(G)) + W;
          // filter
          f = multiply(transpose(F), a);
          Q = quad_form_sym(R, F) + V;
          Q_inv = inverse(Q);
          e = subtract(yi, f);
          A = multiply(multiply(R, F), Q_inv);
          // // // update log-likelihood
          m = add(a, multiply(A, e));
          C = subtract(R, quad_form_sym(Q, transpose(A)));
          lp -= 0.5 * (log_determinant(Q) + trace_quad_form(Q_inv, e));
        }
      }
      return lp;
    }

    template <typename T_y, 
              typename T_F, typename T_G,
              typename T_V, typename T_W
              >
    inline
    typename boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type
    gaussian_dlm_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_F,Eigen::Dynamic,Eigen::Dynamic>& F,
                     const Eigen::Matrix<T_G,Eigen::Dynamic,Eigen::Dynamic>& G,
                     const Eigen::Matrix<T_V,Eigen::Dynamic,Eigen::Dynamic>& V,
                     const Eigen::Matrix<T_W,Eigen::Dynamic,Eigen::Dynamic>& W) {
      return gaussian_dlm_log<false>(y, F, G, V, W);
    }
  }    
}

#endif
