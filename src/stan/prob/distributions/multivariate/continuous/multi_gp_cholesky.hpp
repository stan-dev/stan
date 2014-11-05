#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_GP_CHOLESKY_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_GP_CHOLESKY_HPP

#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/matrix/dot_self.hpp>
#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/row.hpp>
#include <stan/math/matrix/sum.hpp>

namespace stan {
  namespace prob {
    // MultiGPCholesky(y|L,w)   [y.rows() = w.size(), y.cols() = Sigma.rows();
    //                            Sigma symmetric, non-negative, definite]
    /**
     * The log of a multivariate Gaussian Process for the given y, w, and
     * a Cholesky factor L of the kernel matrix Sigma.
     * Sigma = LL', a square, semi-positive definite matrix..  y is a dxN matrix, where each column is a different observation and each
     * row is a different output dimension.  The Gaussian Process is assumed to
     * have a scaled kernel matrix with a different scale for each output dimension.
     * This distribution is equivalent to:
     *    for (i in 1:d) row(y,i) ~ multi_normal(0,(1/w[i])*LL').
     *
     * @param y A dxN matrix
     * @param L The Cholesky decomposition of a kernel matrix
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
    multi_gp_cholesky_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L,
                 const Eigen::Matrix<T_w,Eigen::Dynamic,1>& w) {
      static const std::string function("stan::prob::multi_gp_cholesky_log");
      typedef typename boost::math::tools::promote_args<T_y,T_covar,T_w>::type T_lp;
      T_lp lp(0.0);

      using stan::math::mdivide_left_tri_low;
      using stan::math::dot_self;
      using stan::math::sum;
      using stan::math::log;

      using stan::error_handling::check_size_match;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_positive;

      check_size_match(function, 
                       "Size of random variable (rows y)", y.rows(), 
                       "Size of kernel scales (w)", w.size());
      check_size_match(function, 
                       "Size of random variable", y.cols(),
                       "rows of covariance parameter", L.rows());
      check_finite(function, "Kernel scales", w);
      check_positive(function, "Kernel scales", w);
      check_finite(function, "Random variable", y);
      
      if (y.rows() == 0)
        return lp;
      
      if (include_summand<propto>::value) {
        lp += NEG_LOG_SQRT_TWO_PI * y.rows() * y.cols();
      }

      if (include_summand<propto,T_covar>::value) {
        lp -= L.diagonal().array().log().sum() * y.rows();
      }

      if (include_summand<propto,T_w>::value) {
        lp += 0.5 * y.cols() * sum(log(w));
      }
      
      if (include_summand<propto,T_y,T_w,T_covar>::value) {
        T_lp sum_lp_vec(0.0);
        for (int i = 0; i < y.rows(); i++) {
          Eigen::Matrix<T_y, Eigen::Dynamic, 1> y_row( y.row(i) );
          Eigen::Matrix<typename boost::math::tools::promote_args<T_y,T_covar>::type,
              Eigen::Dynamic, 1> 
            half(mdivide_left_tri_low(L,y_row));
          sum_lp_vec += w(i) * dot_self(half);
        }
        lp -= 0.5*sum_lp_vec;
      }

      return lp;
    }
    
    template <typename T_y, typename T_covar, typename T_w>
    inline
    typename boost::math::tools::promote_args<T_y,T_covar,T_w>::type
    multi_gp_cholesky_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L,
                 const Eigen::Matrix<T_w,Eigen::Dynamic,1>& w) {
      return multi_gp_cholesky_log<false>(y,L,w);
    }
  }    
}

#endif
