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
#include <stan/math/matrix/ldlt.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/math/matrix/quad_form.hpp>
#include <stan/math/matrix/tcrossprod.hpp>

namespace stan {
  namespace prob {
    /**
     * The log of a Gaussian dynamic linear model (GDLM).
     * This distribution is equivalent to, for \f$t = 1:N\f$,
     * \f{eqnarray*}{
     * y_t & \sim N(F' \theta_t, V) \\
     * \theta_t & \sim N(G \theta_{t-1}, W) \\
     * \theta_0 & \sim N(0, diag(10^{6}))
     * \f}
     *
     * @param y A r x T matrix of observations. Rows are variables,
     * columns are observations.
     * @param F A n x r matrix. The design matrix.
     * @param G A n x n matrix. The transition matrix.
     * @param V A r x r matrix. The observation covariance matrix.
     * @param W A n x n matrix. The state covariance matrix.
     * @return The log of the joint density of the GDLM.
     * @throw std::domain_error if a matrix in the Kalman filter is
     * not semi-positive definite.
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
      using stan::math::quad_form_sym;
      // using stan::math::LDLT_factor;
      // using stan::math::trace_inv_quad_form_ldlt;
      // using stan::math::mdivide_right_ldlt;

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
        lp += 0.5 * NEG_LOG_SQRT_TWO_PI * r * T;
      }
      
      if (include_summand<propto,T_y,T_F,T_G,T_V,T_W>::value) {
        // TODO: initial values of m and C should be arguments.
        // however that would create too many args for
        // boost::math::tools::promote_args (max 6)
        // and would require altering function signatures in the
        // parser to accept more arguments.
        Eigen::Matrix<T_lp,Eigen::Dynamic,1> m(n);
        for (int i = 0; i < m.size(); i ++)
          m(i) = 0.0;
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> C(n, n);
        for (int i = 0; i < C.rows(); i ++) {
          for (int j = 0; j < C.cols(); j ++) {          
            if (i == j) {
              C(i, j) = 10e6;
            } else {
              C(i, j) = 0.0;
            }
          }
        }

        Eigen::Matrix<typename boost::math::tools::promote_args<T_y>::type,
                      Eigen::Dynamic, 1> yi(r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, 1> a(n);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> R(n, n);
        Eigen::Matrix<T_lp,Eigen::Dynamic, 1> f(r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> Q(r, r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> Q_inv(r, r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, 1> e(r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> A(n, r);

        for (int i = 0; i < T; ++i) {
          yi = y.col(i);
          // Predict
          a = multiply(G, m);
          R = quad_form_sym(C, transpose(G)) + W;
          // filter
          f = multiply(transpose(F), a);
          Q = quad_form_sym(R, F) + V;
          Q_inv = inverse(Q);
          // LDLT weirdly seems to be going slower.
          // LDLT_factor<T_lp,Eigen::Dynamic,Eigen::Dynamic> ldlt_Q(Q);
          // if (!ldlt_Q.success()) {
          //   std::ostringstream message;
          //   message << "Q is not positive definite (%1%).";
          //   std::string str(message.str());
          //   stan::math::dom_err(function,Q(0,0),"Q",str.c_str(),"",&lp);
          //   return lp;
          // }
          e = subtract(yi, f);
          // A = mdivide_right_ldlt(multiply(R, F), ldlt_Q);
          A = multiply(multiply(R, F), Q_inv);
          m = add(a, multiply(A, e));
          C = subtract(R, quad_form_sym(Q, transpose(A)));
          // lp -= 0.5 * (log_determinant_ldlt(ldlt_Q) + trace_inv_quad_form_ldlt(ldlt_Q, e));
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

    /**
     * If V is a vector, then the sequential version of the Kalman
     * filter is used.
     **/
    template <bool propto,
              typename T_y, 
              typename T_F, typename T_G,
              typename T_V, typename T_W
              >
    typename boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type
    gaussian_dlm_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_F,Eigen::Dynamic,Eigen::Dynamic>& F,
                     const Eigen::Matrix<T_G,Eigen::Dynamic,Eigen::Dynamic>& G,
                     const Eigen::Matrix<T_V,Eigen::Dynamic,1>& V,
                     const Eigen::Matrix<T_W,Eigen::Dynamic,Eigen::Dynamic>& W) {
      static const char* function = "stan::prob::dlm_log(%1%)";
      typedef typename boost::math::tools::promote_args<T_y,T_F,T_G,T_V,T_W>::type T_lp;
      T_lp lp(0.0);
      
      using stan::math::check_not_nan;
      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_cov_matrix;
      using stan::math::check_positive;
      using stan::math::add;
      using stan::math::multiply;
      using stan::math::transpose;
      using stan::math::inverse;
      using stan::math::subtract;
      using stan::math::trace_quad_form;
      using stan::math::quad_form_sym;
      using stan::math::tcrossprod;

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
      if (!check_positive(function, V, "V", &lp))
        return lp;
      if (!check_size_match(function,
                            V.size(), "size of V",
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
        lp += 0.5 * NEG_LOG_SQRT_TWO_PI * r * T;
      }
      
      if (include_summand<propto,T_y,T_F,T_G,T_V,T_W>::value) {
        // TODO: initial values of m and C should be arguments.
        // however that would create too many args for
        // boost::math::tools::promote_args (max 6)
        // and would require altering function signatures in the
        // parser to accept more arguments.
        Eigen::Matrix<T_lp, Eigen::Dynamic, 1> m(n);
        for (int i = 0; i < m.size(); i ++)
          m(i) = 0.0;
        Eigen::Matrix<T_lp, Eigen::Dynamic, Eigen::Dynamic> C(n, n);
        for (int i = 0; i < C.rows(); i ++) {
          for (int j = 0; j < C.cols(); j ++) {          
            if (i == j) {
              C(i, j) = 10e6;
            } else {
              C(i, j) = 0.0;
            }
          }
        }

        T_lp f;
        T_lp Q;
        T_lp Q_inv;
        T_lp e;
        Eigen::Matrix<T_lp, Eigen::Dynamic, 1> A(n);
        
        for (int i = 0; i < y.cols(); ++i) {
          // Predict (reuse m and C)
          m = multiply(G, m);
          C = quad_form_sym(C, transpose(G)) + W;
          for (int j = 0; j < y.rows(); ++j) {
            T_lp yij(y(j, i));
            Eigen::Matrix<T_lp, Eigen::Dynamic, 1> Fj = F.col(j);
            // one step forecast of y(i, j)
            f = dot_product(Fj, m);
            Q = trace_quad_form(C, Fj) + V(j);
            Q_inv = 1.0 / Q;
            // posterior after observing y(i, j)
            e = yij - f;
            // A = multiply(multiply(C, Fj), Q_inv);            
            A = C * Fj * Q_inv;
            m += A * e;
            // tcrossprod throws an error (ambiguous)
            // C -= Q * tcrossprod(A);
            C -= Q * transpose(A) * A;
            lp -= 0.5 * (abs(Q) + pow(e, 2) * Q_inv);
          }
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
                     const Eigen::Matrix<T_V,Eigen::Dynamic,1>& V,
                     const Eigen::Matrix<T_W,Eigen::Dynamic,Eigen::Dynamic>& W) {
      return gaussian_dlm_log<false>(y, F, G, V, W);
    }
  }    

}

#endif
