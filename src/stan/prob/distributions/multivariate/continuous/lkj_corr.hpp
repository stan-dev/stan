#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_CORR_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_CORR_HPP

#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/continuous/beta.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/transform.hpp>

namespace stan {
  namespace prob {

    template <typename T_shape>
    T_shape do_lkj_constant(const T_shape& eta, const unsigned int& K) {

      using stan::math::sum;
      using stan::math::lgamma;

      // Lewandowski, Kurowicka, and Joe (2009) theorem 5
      T_shape constant;
      const int Km1 = K - 1;
      if (eta == 1.0) {
        // C++ integer division is appropriate in this block
        Eigen::VectorXd numerator( Km1 / 2 );
        for(size_t k = 1; k <= numerator.rows(); k++)
          numerator(k-1) = lgamma(2 * k);
        constant = sum(numerator);
        if ( (K % 2) == 1 ) constant += 0.25 * (K * K - 1) * LOG_PI -
          0.25 * (Km1 * Km1) * LOG_TWO - Km1 * lgamma( (K + 1) / 2);
        else constant += 0.25 * K * (K - 2) * LOG_PI +
          0.25 * (3 * K * K - 4 * K) * LOG_TWO +
          K * lgamma(K / 2) - Km1 * lgamma(K);
      }
      else {
        constant = -Km1 * lgamma(eta + 0.5 * Km1);
        for (size_t k = 1; k <= Km1; k++)
          constant += 0.5 * k * LOG_PI + lgamma(eta + 0.5 * (Km1 - k));
      }
      return constant;
    }

    // LKJ_Corr(L|eta) [ L Cholesky factor of correlation matrix
    //                  eta > 0; eta == 1 <-> uniform]
    template <bool propto,
              typename T_covar, typename T_shape>
    typename boost::math::tools::promote_args<T_covar, T_shape>::type
    lkj_corr_cholesky_log(
             const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L, 
             const T_shape& eta) {

      static const std::string function("stan::prob::lkj_corr_cholesky_log");

      using boost::math::tools::promote_args;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_lower_triangular;
      using stan::math::sum;
      
      typename promote_args<T_covar,T_shape>::type lp(0.0);
      check_positive(function, "Shape parameter", eta);
      check_lower_triangular(function, "Random variable", L);

      const unsigned int K = L.rows();
      if (K == 0)
        return 0.0;
            
      if (include_summand<propto,T_shape>::value) 
        lp += do_lkj_constant(eta, K);
      if (include_summand<propto,T_covar,T_shape>::value) {
        const int Km1 = K - 1;
        Eigen::Matrix<T_covar,Eigen::Dynamic,1> log_diagonals =
          L.diagonal().tail(Km1).array().log();
        Eigen::Matrix<T_covar,Eigen::Dynamic,1> values(Km1);
        for (size_t k = 0; k < Km1; k++)
          values(k) = (Km1 - k - 1) * log_diagonals(k);
        if ( (eta == 1.0) &&
            stan::is_constant<typename stan::scalar_type<T_shape> >::value) {
            lp += sum(values);
            return(lp);
        }
        values += (2.0 * eta - 2.0) * log_diagonals;
        lp += sum(values);
      }
      
      return lp;
    }

    template <typename T_covar, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_covar, T_shape>::type
    lkj_corr_cholesky_log(
             const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L, 
             const T_shape& eta) {
      return lkj_corr_cholesky_log<false>(L,eta);
    }

    // LKJ_Corr(y|eta) [ y correlation matrix (not covariance matrix)
    //                  eta > 0; eta == 1 <-> uniform]
    template <bool propto,
              typename T_y, typename T_shape>
    typename boost::math::tools::promote_args<T_y, T_shape>::type
    lkj_corr_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y, 
                 const T_shape& eta) {
      static const std::string function("stan::prob::lkj_corr_log");

      using stan::error_handling::check_size_match;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_corr_matrix;
      using stan::math::sum;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_shape>::type lp(0.0);
      check_positive(function, "Shape parameter", eta);
      check_size_match(function, 
                       "Rows of correlation matrix", y.rows(), 
                       "columns of correlation matrix", y.cols());
      check_not_nan(function, "Correlation matrix", y);
      check_corr_matrix(function, "Correlation matrix", y);
      
      const unsigned int K = y.rows();
      if (K == 0)
        return 0.0;

      if (include_summand<propto,T_shape>::value)
        lp += do_lkj_constant(eta, K);

      if ( (eta == 1.0) &&
          stan::is_constant<typename stan::scalar_type<T_shape> >::value )
        return lp;

      if (!include_summand<propto,T_y,T_shape>::value)
          return lp;

      Eigen::Matrix<T_y,Eigen::Dynamic,1> values =
        y.ldlt().vectorD().array().log().matrix();
      lp += (eta - 1.0) * sum(values);
      return lp;
    }

    template <typename T_y, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_y, T_shape>::type
    lkj_corr_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y, 
                 const T_shape& eta) {
      return lkj_corr_log<false>(y,eta);
    }

    template <class RNG>
    inline Eigen::MatrixXd
    lkj_corr_cholesky_rng(const size_t K,
                          const double eta,
                          RNG& rng) {
      static const std::string function("stan::prob::lkj_corr_cholesky_rng");

      using stan::error_handling::check_positive;
      
      check_positive(function, "Shape parameter", eta);

      Eigen::ArrayXd CPCs( (K * (K - 1)) / 2 );
      double alpha = eta + 0.5 * (K - 1);
      unsigned int count = 0;
      for (size_t i = 0; i < (K - 1); i++) {
        alpha -= 0.5;
        for (size_t j = i + 1; j < K; j++) {
          CPCs(count) = 2.0 * stan::prob::beta_rng(alpha,alpha,rng) - 1.0;
          count++;
        }
      }
      return stan::prob::read_corr_L(CPCs, K);
    }

    template <class RNG>
    inline Eigen::MatrixXd
    lkj_corr_rng(const size_t K,
                 const double eta,
                 RNG& rng) {

      static const std::string function("stan::prob::lkj_corr_rng");

      using stan::error_handling::check_positive;
      
      check_positive(function, "Shape parameter", eta);

      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(
                  lkj_corr_cholesky_rng(K, eta, rng) );
    }

  }
}
#endif
