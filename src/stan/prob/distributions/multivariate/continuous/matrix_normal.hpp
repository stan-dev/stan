#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MATRIX_NORMAL_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MATRIX_NORMAL_HPP

#include <stan/error_handling/matrix/check_ldlt_factor.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/log_determinant.hpp>
#include <stan/math/matrix/log_determinant_ldlt.hpp>
#include <stan/math/matrix/subtract.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/math/matrix/trace_gen_quad_form.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

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
      static const std::string function("stan::prob::matrix_normal_prec_log");
      typename boost::math::tools::promote_args<T_y,T_Mu,T_Sigma,T_D>::type lp(0.0);
      
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_symmetric;
      using stan::error_handling::check_size_match;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_finite;
      using stan::math::trace_gen_quad_form;
      using stan::math::log_determinant_ldlt;
      using stan::math::subtract;
      using stan::math::LDLT_factor;
      using stan::error_handling::check_ldlt_factor;
      
      check_size_match(function, 
                       "Rows of Sigma", Sigma.rows(), 
                       "columns of Sigma", Sigma.cols());
      check_positive(function, "Sigma rows", Sigma.rows());
      check_finite(function, "Sigma", Sigma);
      check_symmetric(function, "Sigma", Sigma);
      
      LDLT_factor<T_Sigma,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function, "LDLT_Factor of Sigma", ldlt_Sigma);
      check_size_match(function, 
                       "Rows of D", D.rows(), 
                       "Columns of D", D.cols());
      check_positive(function, "D rows", D.rows());
      check_finite(function, "D", D);
      check_symmetric(function, "Sigma", D);
      
      LDLT_factor<T_D,Eigen::Dynamic,Eigen::Dynamic> ldlt_D(D);
      check_ldlt_factor(function, "LDLT_Factor of D", ldlt_D);
      check_size_match(function, 
                       "Rows of random variable", y.rows(),
                       "Rows of location parameter", Mu.rows());
      check_size_match(function, 
                       "Columns of random variable", y.cols(),
                       "Columns of location parameter", Mu.cols());
      check_size_match(function, 
                       "Rows of random variable", y.rows(),
                       "Rows of Sigma", Sigma.rows());
      check_size_match(function, 
                       "Columns of random variable", y.cols(),
                       "Rows of D", D.rows());
      check_finite(function, "Location parameter", Mu);
      check_finite(function, "Random variable", y);
      
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
