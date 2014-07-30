#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MATRIX_NORMAL_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MATRIX_NORMAL_HPP

#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/meta/traits.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/math/error_handling/matrix/check_ldlt_factor.hpp>
#include <stan/math/matrix/log_determinant_ldlt.hpp>

namespace stan {
  namespace prob {
    /**
     * The log of the matrix normal density for the given y, mu, Sigma and D
     * where Sigma and D are given as precision matrices, not covariance matrices.
     *
     * @param y An mxn matrix.
     * @param Mu The mean matrix.
     * @param Sigma The mxm inverse covariance matrix (i.e., the precision matrix) of the 
     * rows of y.
     * @param D The nxn inverse covariance matrix (i.e., the precision matrix) of the 
     * columns of y.
     * @return The log of the matrix normal density.
     * @throw std::domain_error if Sigma or D are not square, not symmetric, 
     * or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_Mu Type of location.
     * @tparam T_Sigma Type of Sigma.
     * @tparam T_D Type of D.
     */
    template <bool propto,
    typename T_y, typename T_Mu, typename T_Sigma, typename T_D>
    typename boost::math::tools::promote_args<T_y,T_Mu,T_Sigma,T_D>::type
    matrix_normal_prec_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                           const Eigen::Matrix<T_Mu,Eigen::Dynamic,Eigen::Dynamic>& Mu,
                           const Eigen::Matrix<T_Sigma,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                           const Eigen::Matrix<T_D,Eigen::Dynamic,Eigen::Dynamic>& D) {
      static const char* function = "stan::prob::matrix_normal_prec_log(%1%)";
      typename boost::math::tools::promote_args<T_y,T_Mu,T_Sigma,T_D>::type lp(0.0);
      
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
      using stan::math::check_size_match;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::trace_gen_quad_form;
      using stan::math::log_determinant_ldlt;
      using stan::math::subtract;
      using stan::math::LDLT_factor;
      using stan::math::check_ldlt_factor;
      
      check_size_match(function, 
                       Sigma.rows(), "Rows of Sigma",
                       Sigma.cols(), "columns of Sigma",
                       &lp);
      check_positive(function, Sigma.rows(), "Sigma rows", &lp);
      check_finite(function, Sigma, "Sigma", &lp);
      check_symmetric(function, Sigma, "Sigma", &lp);
      
      LDLT_factor<T_Sigma,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function,ldlt_Sigma,"LDLT_Factor of Sigma",&lp);
      check_size_match(function, 
                       D.rows(), "Rows of D",
                       D.cols(), "Columns of D",
                       &lp);
      check_positive(function, D.rows(), "D rows", &lp);
      check_finite(function, D, "D", &lp);
      check_symmetric(function, D, "Sigma", &lp);
      
      LDLT_factor<T_D,Eigen::Dynamic,Eigen::Dynamic> ldlt_D(D);
      check_ldlt_factor(function,ldlt_D,"LDLT_Factor of D",&lp);
      check_size_match(function, 
                       y.rows(), "Rows of random variable",
                       Mu.rows(), "Rows of location parameter",
                       &lp);
      check_size_match(function, 
                       y.cols(), "Columns of random variable",
                       Mu.cols(), "Columns of location parameter",
                       &lp);
      check_size_match(function, 
                       y.rows(), "Rows of random variable",
                       Sigma.rows(), "Rows of Sigma",
                       &lp);
      check_size_match(function, 
                       y.cols(), "Columns of random variable",
                       D.rows(), "Rows of D",
                       &lp);
      check_finite(function, Mu, "Location parameter", &lp);
      check_finite(function, y, "Random variable", &lp);
      
      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * y.cols() * y.rows();
      
      if (include_summand<propto,T_Sigma>::value) {
        lp += log_determinant_ldlt(ldlt_Sigma) * (0.5 * y.rows());
      }

      if (include_summand<propto,T_D>::value) {
        lp += log_determinant_ldlt(ldlt_D) * (0.5 * y.cols());
      }
      
      if (include_summand<propto,T_y,T_Mu,T_Sigma,T_D>::value) {
        lp -= 0.5 * trace_gen_quad_form(D,Sigma,subtract(y,Mu));
      }
      return lp;      
    }

    template <typename T_y, typename T_Mu, typename T_Sigma, typename T_D>
    typename boost::math::tools::promote_args<T_y,T_Mu,T_Sigma,T_D>::type
    matrix_normal_prec_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                           const Eigen::Matrix<T_Mu,Eigen::Dynamic,Eigen::Dynamic>& Mu,
                           const Eigen::Matrix<T_Sigma,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                           const Eigen::Matrix<T_D,Eigen::Dynamic,Eigen::Dynamic>& D) {
      return matrix_normal_prec_log<false>(y,Mu,Sigma,D);
    }
  }
}

#endif
