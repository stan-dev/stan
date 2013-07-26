#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__GAUSSIAN_DLM_OBS_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__GAUSSIAN_DLM_OBS_HPP__

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
#include <stan/agrad/rev/matrix/to_var.hpp>

#include <stan/math/matrix/add.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/inverse_spd.hpp>
#include <stan/math/matrix/ldlt.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/log_determinant_spd.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/quad_form.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/tcrossprod.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/math/matrix/transpose.hpp>

/*
  TODO: time-varying system matrices
  TODO: use sequential processing even for non-diagonal obs
  covariance.
  TODO: add constant terms in observation.
 */

namespace stan {
  namespace prob {
    /**
     * The log of a Gaussian dynamic linear model (GDLM).
     * This distribution is equivalent to, for \f$t = 1:T\f$,
     * \f{eqnarray*}{
     * y_t & \sim N(F' \theta_t, V) \\
     * \theta_t & \sim N(G \theta_{t-1}, W) \\
     * \theta_0 & \sim N(m_0, C_0)
     * \f}
     *
     * If V is a vector, then the Kalman filter is applied
     * sequentially.
     *
     * @param y A r x T matrix of observations. Rows are variables,
     * columns are observations.
     * @param F A n x r matrix. The design matrix.
     * @param G A n x n matrix. The transition matrix.
     * @param V A r x r matrix. The observation covariance matrix.
     * @param W A n x n matrix. The state covariance matrix.
     * @param m0 A n x 1 matrix. The mean vector of the distribution
     * of the initial state.
     * @param C0 A n x n matrix. The covariance matrix of the
     * distribution of the initial state.
     * @return The log of the joint density of the GDLM.
     * @throw std::domain_error if a matrix in the Kalman filter is
     * not positive semi-definite.
     * @tparam T_y Type of scalar.
     * @tparam T_F Type of design matrix.
     * @tparam T_G Type of transition matrix.
     * @tparam T_V Type of observation covariance matrix.
     * @tparam T_W Type of state covariance matrix.
     * @tparam T_m0 Type of initial state mean vector.
     * @tparam T_C0 Type of initial state covariance matrix.
     */
    template <bool propto,
              typename T_y,
              typename T_F, typename T_G,
              typename T_V, typename T_W,
              typename T_m0, typename T_C0
              >
    typename return_type<
      T_y,
      typename return_type<T_F,T_G,T_V,T_W,T_m0,T_C0>::type >::type
    gaussian_dlm_obs_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_F,Eigen::Dynamic,Eigen::Dynamic>& F,
                     const Eigen::Matrix<T_G,Eigen::Dynamic,Eigen::Dynamic>& G,
                     const Eigen::Matrix<T_V,Eigen::Dynamic,Eigen::Dynamic>& V,
                     const Eigen::Matrix<T_W,Eigen::Dynamic,Eigen::Dynamic>& W,
                     const Eigen::Matrix<T_m0,Eigen::Dynamic,1>& m0,
                     const Eigen::Matrix<T_C0,Eigen::Dynamic,Eigen::Dynamic>& C0) {
      static const char* function = "stan::prob::gaussian_dlm_obs_log(%1%)";
      typedef typename return_type<
        T_y,
        typename return_type<T_F,T_G,T_V,T_W,T_m0,T_C0>::type  >::type T_lp;
      T_lp lp(0.0);

      using stan::math::add;
      using stan::math::check_cov_matrix;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_size_match;
      using stan::math::check_spsd_matrix;
      using stan::math::inverse_spd;
      using stan::math::log_determinant_spd;
      using stan::math::multiply;
      using stan::math::quad_form_sym;
      using stan::math::subtract;
      using stan::math::trace_quad_form;
      using stan::math::transpose;

      int r = y.rows(); // number of variables
      int T = y.cols(); // number of observations
      int n = G.rows(); // number of states

      // check y
      if (!check_finite(function, y, "y", &lp))
        return lp;
      if (!check_not_nan(function, y, "y", &lp))
        return lp;
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
      if (!check_finite(function, F, "F", &lp))
        return lp;
      if (!check_not_nan(function, F, "F", &lp))
        return lp;
      // check G
      if (!check_size_match(function,
                            G.rows(), "rows of G",
                            G.cols(), "columns of G",
                            &lp))
        return lp;
      if (!check_finite(function, G, "G", &lp))
        return lp;
      if (!check_not_nan(function, G, "G", &lp))
        return lp;
      // check V
      if (!check_size_match(function,
                            V.rows(), "rows of V",
                            y.rows(), "rows of y",
                            &lp))
        return lp;
      // TODO: incorporate support for infinite V
      if (!check_finite(function, V, "V", &lp))
        return lp;
      if (!check_not_nan(function, V, "V", &lp))
        return lp;
      if (!check_spsd_matrix(function, V, "V", &lp))
        return lp;
      // check W
      if (!check_size_match(function,
                            W.rows(), "rows of W",
                            G.rows(), "rows of G",
                            &lp))
        return lp;
      // TODO: incorporate support for infinite W
      if (!check_finite(function, W, "W", &lp))
        return lp;
      if (!check_not_nan(function, W, "W", &lp))
        return lp;
      if (!check_spsd_matrix(function, W, "W", &lp))
        return lp;
      // check m0
      if (!check_size_match(function,
                            m0.size(), "size of m0",
                            G.rows(), "rows of G",
                            &lp))
        return lp;
      if (!check_finite(function, m0, "m0", &lp))
        return lp;
      if (!check_not_nan(function, m0, "m0", &lp))
        return lp;
      // check C0
      if (!check_size_match(function,
                            C0.rows(), "rows of C0",
                            G.rows(), "rows of G",
                            &lp))
        return lp;
      if (!check_cov_matrix(function, C0, "C0", &lp))
        return lp;
      if (!check_finite(function, C0, "C0", &lp))
        return lp;
      if (!check_not_nan(function, C0, "C0", &lp))
        return lp;

      if (y.cols() == 0 || y.rows() == 0)
        return lp;

      if (include_summand<propto>::value) {
        lp -= 0.5 * LOG_TWO_PI * r * T;
      }

      if (include_summand<propto,T_y,T_F,T_G,T_V,T_W,T_m0,T_C0>::value) {
        Eigen::Matrix<T_lp,Eigen::Dynamic, 1> m(n);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> C(n, n);

        // TODO: how to recast matrices
        for (int i = 0; i < m0.size(); i ++ ) {
          m(i) = m0(i);
        }
        for (int i = 0; i < C0.rows(); i ++ ) {
          for (int j = 0; j < C0.cols(); j ++ ) {
            C(i, j) = C0(i, j);
          }
        }

        Eigen::Matrix<typename return_type<T_y>::type,
                      Eigen::Dynamic, 1> yi(r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, 1> a(n);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> R(n, n);
        Eigen::Matrix<T_lp,Eigen::Dynamic, 1> f(r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> Q(r, r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> Q_inv(r, r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, 1> e(r);
        Eigen::Matrix<T_lp,Eigen::Dynamic, Eigen::Dynamic> A(n, r);

        for (int i = 0; i < y.cols(); i++) {
          yi = y.col(i);
          // // Predict state
          // a_t = G_t m_{t-1}
          a = multiply(G, m);
          // R_t = G_t C_{t-1} G_t' + W_t
          R = add(quad_form_sym(C, transpose(G)), W);
          // // predict observation
          // f_t = F_t' a_t
          f = multiply(transpose(F), a);
          // Q_t = F'_t R_t F_t + V_t
          Q = add(quad_form_sym(R, F), V);
          Q_inv = inverse_spd(Q);
          // // filtered state
          // e_t = y_t - f_t
          e = subtract(yi, f);
          // A_t = R_t F_t Q^{-1}_t
          A = multiply(multiply(R, F), Q_inv);
          // m_t = a_t + A_t e_t
          m = add(a, multiply(A, e));
          // C = R_t - A_t Q_t A_t'
          C = subtract(R, quad_form_sym(Q, transpose(A)));
          lp -= 0.5 * (log_determinant_spd(Q) + trace_quad_form(Q_inv, e));
        }
      }
      return lp;
    }

    template <typename T_y,
              typename T_F, typename T_G,
              typename T_V, typename T_W,
              typename T_m0, typename T_C0
              >
    inline
    typename return_type<
      T_y,
      typename return_type<T_F,T_G,T_V,T_W,T_m0,T_C0>::type >::type
    gaussian_dlm_obs_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_F,Eigen::Dynamic,Eigen::Dynamic>& F,
                     const Eigen::Matrix<T_G,Eigen::Dynamic,Eigen::Dynamic>& G,
                     const Eigen::Matrix<T_V,Eigen::Dynamic,Eigen::Dynamic>& V,
                     const Eigen::Matrix<T_W,Eigen::Dynamic,Eigen::Dynamic>& W,
                     const Eigen::Matrix<T_m0,Eigen::Dynamic,1>& m0,
                     const Eigen::Matrix<T_C0,Eigen::Dynamic,Eigen::Dynamic>& C0) {
      return gaussian_dlm_obs_log<false>(y, F, G, V, W, m0, C0);
    }

    /**
     * The log of a Gaussian dynamic linear model (GDLM) with
     * uncorrelated observation disturbances.
     * This distribution is equivalent to, for \f$t = 1:T\f$,
     * \f{eqnarray*}{
     * y_t & \sim N(F' \theta_t, diag(V)) \\
     * \theta_t & \sim N(G \theta_{t-1}, W) \\
     * \theta_0 & \sim N(m_0, C_0)
     * \f}
     *
     * If V is a vector, then the Kalman filter is applied
     * sequentially.
     *
     * @param y A r x T matrix of observations. Rows are variables,
     * columns are observations.
     * @param F A n x r matrix. The design matrix.
     * @param G A n x n matrix. The transition matrix.
     * @param V A size r vector. The diagonal of the observation
     * covariance matrix.
     * @param W A n x n matrix. The state covariance matrix.
     * @param m0 A n x 1 matrix. The mean vector of the distribution
     * of the initial state.
     * @param C0 A n x n matrix. The covariance matrix of the
     * distribution of the initial state.
     * @return The log of the joint density of the GDLM.
     * @throw std::domain_error if a matrix in the Kalman filter is
     * not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_F Type of design matrix.
     * @tparam T_G Type of transition matrix.
     * @tparam T_V Type of observation variances
     * @tparam T_W Type of state covariance matrix.
     * @tparam T_m0 Type of initial state mean vector.
     * @tparam T_C0 Type of initial state covariance matrix.
     */
    template <bool propto,
              typename T_y,
              typename T_F, typename T_G,
              typename T_V, typename T_W,
              typename T_m0, typename T_C0
              >
    typename return_type<
      T_y,
      typename return_type<T_F,T_G,T_V,T_W,T_m0,T_C0>::type >::type
    gaussian_dlm_obs_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_F,Eigen::Dynamic,Eigen::Dynamic>& F,
                     const Eigen::Matrix<T_G,Eigen::Dynamic,Eigen::Dynamic>& G,
                     const Eigen::Matrix<T_V,Eigen::Dynamic,1>& V,
                     const Eigen::Matrix<T_W,Eigen::Dynamic,Eigen::Dynamic>& W,
                     const Eigen::Matrix<T_m0,Eigen::Dynamic,1>& m0,
                     const Eigen::Matrix<T_C0,Eigen::Dynamic,Eigen::Dynamic>& C0) {
      static const char* function = "stan::prob::gaussian_dlm_obs_log(%1%)";
      typedef typename return_type<
        T_y,
        typename return_type<T_F,T_G,T_V,T_W,T_m0,T_C0>::type  >::type T_lp;
      T_lp lp(0.0);

      using stan::math::add;
      using stan::math::check_cov_matrix;
      using stan::math::check_finite;
      using stan::math::check_nonnegative;
      using stan::math::check_not_nan;
      using stan::math::check_size_match;
      using stan::math::check_spsd_matrix;
      using stan::math::dot_product;
      using stan::math::multiply;
      using stan::math::quad_form_sym;
      using stan::math::subtract;
      using stan::math::tcrossprod;
      using stan::math::trace_quad_form;
      using stan::math::transpose;

      int r = y.rows(); // number of variables
      int T = y.cols(); // number of observations
      int n = G.rows(); // number of states

      // check y
      if (!check_finite(function, y, "y", &lp))
        return lp;
      if (!check_not_nan(function, y, "y", &lp))
        return lp;
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
      if (!check_finite(function, F, "F", &lp))
        return lp;
      if (!check_not_nan(function, F, "F", &lp))
        return lp;
      // check G
      if (!check_size_match(function,
                            G.rows(), "rows of G",
                            G.cols(), "columns of G",
                            &lp))
        return lp;
      if (!check_finite(function, G, "G", &lp))
        return lp;
      if (!check_not_nan(function, G, "G", &lp))
        return lp;
      // check V
      if (!check_nonnegative(function, V, "V", &lp))
        return lp;
      if (!check_size_match(function,
                            V.size(), "size of V",
                            y.rows(), "rows of y",
                            &lp))
        return lp;
      // TODO: support infinite V
      if (!check_finite(function, V, "V", &lp))
        return lp;
      if (!check_not_nan(function, V, "V", &lp))
        return lp;
      // check W
      if (!check_spsd_matrix(function, W, "W", &lp))
        return lp;
      if (!check_size_match(function,
                            W.rows(), "rows of W",
                            G.rows(), "rows of G",
                            &lp))
        return lp;
      // TODO: support infinite W
      if (!check_finite(function, W, "W", &lp))
        return lp;
      if (!check_not_nan(function, W, "W", &lp))
        return lp;
      // check m0
      if (!check_size_match(function,
                            m0.size(), "size of m0",
                            G.rows(), "rows of G",
                            &lp))
        return lp;
      if (!check_finite(function, m0, "m0", &lp))
        return lp;
      if (!check_not_nan(function, m0, "m0", &lp))
        return lp;
      // check C0
      if (!check_cov_matrix(function, C0, "C0", &lp))
        return lp;
      if (!check_size_match(function,
                            C0.rows(), "rows of C0",
                            G.rows(), "rows of G",
                            &lp))
        return lp;
      if (!check_finite(function, C0, "C0", &lp))
        return lp;
      if (!check_not_nan(function, C0, "C0", &lp))
        return lp;

      if (y.cols() == 0 || y.rows() == 0)
        return lp;

      if (include_summand<propto>::value) {
        lp += 0.5 * NEG_LOG_TWO_PI * r * T;
      }

      if (include_summand<propto,T_y,T_F,T_G,T_V,T_W,T_m0,T_C0>::value) {
        T_lp f;
        T_lp Q;
        T_lp Q_inv;
        T_lp e;
        Eigen::Matrix<T_lp, Eigen::Dynamic, 1> A(n);
        Eigen::Matrix<T_lp, Eigen::Dynamic, 1> Fj(n);
        Eigen::Matrix<T_lp, Eigen::Dynamic, 1> m(n);
        Eigen::Matrix<T_lp, Eigen::Dynamic, Eigen::Dynamic> C(n, n);

        // TODO: how to recast matrices
        for (int i = 0; i < m0.size(); i ++ ) {
          m(i) = m0(i);
        }
        for (int i = 0; i < C0.rows(); i ++ ) {
          for (int j = 0; j < C0.cols(); j ++ ) {
            C(i, j) = C0(i, j);
          }
        }

        for (int i = 0; i < y.cols(); i++) {
          // Predict state
          // reuse m and C instead of using a and R
          m = multiply(G, m);
          C = add(quad_form_sym(C, transpose(G)), W);
          for (int j = 0; j < y.rows(); ++j) {
            // predict observation
            T_lp yij(y(j, i));
            // dim Fj = (n, 1)
            for (int k = 0; k < F.rows(); ++k) {
              Fj(k) = F(k, j);
            }
            // // f_{t,i} = F_{t,i}' m_{t,i-1}
            f = dot_product(Fj, m);
            Q = trace_quad_form(C, Fj) + V(j);
            Q_inv = 1.0 / Q;
            // // filtered observation
            // // e_{t,i} = y_{t,i} - f_{t,i}
            e = yij - f;
            // // A_{t,i} = C_{t,i-1} F_{t,i} Q_{t,i}^{-1}
            A = multiply(multiply(C, Fj), Q_inv);
            // // m_{t,i} = m_{t,i-1} + A_{t,i} e_{t,i}
            m += multiply(A, e);
            // // c_{t,i} = C_{t,i-1} - Q_{t,i} A_{t,i} A_{t,i}'
            // // // tcrossprod throws an error (ambiguous)
            // C = subtract(C, multiply(Q, tcrossprod(A)));
            C -= multiply(Q, multiply(A, transpose(A)));
            C = 0.5 * add(C, transpose(C));
            lp -= 0.5 * (log(Q) + pow(e, 2) * Q_inv);
          }
        }
      }
      return lp;
    }

    template <typename T_y,
              typename T_F, typename T_G,
              typename T_V, typename T_W,
              typename T_m0, typename T_C0
              >
    inline
    typename return_type<
      T_y,
      typename return_type<T_F,T_G,T_V,T_W,T_m0,T_C0>::type
      >::type
    gaussian_dlm_obs_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                     const Eigen::Matrix<T_F,Eigen::Dynamic,Eigen::Dynamic>& F,
                     const Eigen::Matrix<T_G,Eigen::Dynamic,Eigen::Dynamic>& G,
                     const Eigen::Matrix<T_V,Eigen::Dynamic,1>& V,
                     const Eigen::Matrix<T_W,Eigen::Dynamic,Eigen::Dynamic>& W,
                     const Eigen::Matrix<T_m0,Eigen::Dynamic,1>& m0,
                     const Eigen::Matrix<T_C0,Eigen::Dynamic,Eigen::Dynamic>& C0) {
      return gaussian_dlm_obs_log<false>(y, F, G, V, W, m0, C0);
    }
  }

}

#endif
