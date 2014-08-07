#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_GP_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_GP_HPP

#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/meta/traits.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/sum.hpp>

#include <stan/math/matrix/log_determinant_ldlt.hpp>
#include <stan/math/matrix/trace_gen_inv_quad_form_ldlt.hpp>
#include <stan/math/error_handling/matrix/check_ldlt_factor.hpp>

namespace stan {
  namespace prob {
    // MultiGP(y|Sigma,w)   [y.rows() = w.size(), y.cols() = Sigma.rows();
    //                            Sigma symmetric, non-negative, definite]
    /**
     * The log of a multivariate Gaussian Process for the given y, Sigma, and
     * w.  y is a dxN matrix, where each column is a different observation and each
     * row is a different output dimension.  The Gaussian Process is assumed to
     * have a scaled kernel matrix with a different scale for each output dimension.
     * This distribution is equivalent to:
     *    for (i in 1:d) row(y,i) ~ multi_normal(0,(1/w[i])*Sigma).
     *
     * @param y A dxN matrix
     * @param Sigma The NxN kernel matrix
     * @param w A d-dimensional vector of positve inverse scale parameters for each output.
     * @return The log of the multivariate GP density.
     * @throw std::domain_error if Sigma is not square, not symmetric, 
     * or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_covar Type of kernel.
     * @tparam T_w Type of weight.
     */
    template <bool propto,
              typename T_y, typename T_covar, typename T_w>
    typename boost::math::tools::promote_args<T_y,T_covar,T_w>::type
    multi_gp_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                 const Eigen::Matrix<T_w,Eigen::Dynamic,1>& w) {
      static const char* function = "stan::prob::multi_gp_log(%1%)";
      typedef typename boost::math::tools::promote_args<T_y,T_covar,T_w>::type T_lp;
      T_lp lp(0.0);
      
      using stan::math::sum;
      using stan::math::log;
      using stan::math::LDLT_factor;
      using stan::math::log_determinant_ldlt;
      using stan::math::trace_gen_inv_quad_form_ldlt;

      using stan::math::check_size_match;
      using stan::math::check_positive_finite;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_symmetric;
      using stan::math::check_ldlt_factor;
      using stan::math::check_not_nan;

      check_size_match(function, 
                       Sigma.rows(), "Rows of kernel matrix",
                       Sigma.cols(), "columns of kernel matrix",
                       &lp);
      check_positive(function, Sigma.rows(), "Kernel rows", &lp);
      check_finite(function, Sigma, "Kernel", &lp);
      check_symmetric(function, Sigma, "Kernel", &lp);
      
      LDLT_factor<T_covar,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function,ldlt_Sigma,"LDLT_Factor of Sigma",&lp);

      check_size_match(function, 
                       y.rows(), "Size of random variable (rows y)",
                       w.size(), "Size of kernel scales (w)",
                       &lp);
      check_size_match(function, 
                       y.cols(), "Size of random variable",
                       Sigma.rows(), "rows of covariance parameter",
                       &lp);
      check_positive_finite(function, w, "Kernel scales", &lp);
      check_finite(function, y, "Random variable", &lp);
      
      if (y.rows() == 0)
        return lp;
      
      if (include_summand<propto>::value) {
        lp += NEG_LOG_SQRT_TWO_PI * y.rows() * y.cols();
      }

      if (include_summand<propto,T_covar>::value) {
        lp -= 0.5 * log_determinant_ldlt(ldlt_Sigma) * y.rows();
      }

      if (include_summand<propto,T_w>::value) {
        lp += (0.5 * y.cols()) * sum(log(w));
      }
      
      if (include_summand<propto,T_y,T_w,T_covar>::value) {
        Eigen::Matrix<T_w,Eigen::Dynamic,Eigen::Dynamic> w_mat(w.asDiagonal());
        Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> yT(y.transpose());
        lp -= 0.5 * trace_gen_inv_quad_form_ldlt(w_mat,ldlt_Sigma,yT);
      }

      return lp;
    }
    
    template <typename T_y, typename T_covar, typename T_w>
    inline
    typename boost::math::tools::promote_args<T_y,T_covar,T_w>::type
    multi_gp_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                 const Eigen::Matrix<T_w,Eigen::Dynamic,1>& w) {
      return multi_gp_log<false>(y,Sigma,w);
    }
  }    
}

#endif
