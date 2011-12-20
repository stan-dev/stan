#ifndef __STAN__PROB__DISTRIBUTIONS_MULTI_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS_MULTI_NORMAL_HPP__

#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

#include <stan/meta/traits.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    using Eigen::Matrix;
    using Eigen::Dynamic;

    // MultiNormal(y|mu,Sigma)   [y.rows() = mu.rows() = Sigma.rows();
    //                            y.cols() = mu.cols() = 0;
    //                            Sigma symmetric, non-negative, definite]
    /**
     * The log of the multivariate normal density for the given y, mu, and
     * variance matrix. 
     * The variance matrix, Sigma, must be size d x d, symmetric,
     * and semi-positive definite. Dimension, d, is implicit.
     *
     * \f{eqnarray*}{
     y &\sim& \mbox{\sf{N}} (\mu, \Sigma) \\
     \log (p (y \,|\, \mu, \Sigma) ) &=& \log \left( (2 \pi)^{-d/2} \left| \Sigma \right|^{-1/2} \times \exp \left(-\frac{1}{2}(y - \mu)^T \Sigma^{-1} (y - \mu) \right) \right) \\
     &=& -\frac{d}{2}\log (2 \pi) - \frac{1}{2} \log (\det(\Sigma)) - \frac{1}{2} (y - \mu)^T \Sigma^{-1} (y - \mu)
     \f}
     * 
     * 
     * @param y A scalar vector
     * @param mu The mean vector of the multivariate normal distribution.
     * @param Sigma The variance matrix of the multivariate normal distribution
     * @return The log of the multivariate normal density.
     * @throw std::domain_error if Sigma is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_covar Type of scale.
     */
    template <bool propto = false, 
	      typename T_y, typename T_loc, typename T_covar, 
	      class Policy = policy<> > 
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(const Matrix<T_y,Dynamic,1>& y,
		     const Matrix<T_loc,Dynamic,1>& mu,
		     const Matrix<T_covar,Dynamic,Dynamic>& Sigma,
		     const Policy& = Policy()) {
      static const char* function = "stan::prob::multi_normal_log<%1%>(%1%)";
      
      double result;
      if (!stan::prob::check_size_match(function,y.size(),mu.size(),&result, Policy()))
	return result;
      if (!stan::prob::check_size_match(function,y.size(),Sigma.rows(),&result, Policy()))
	return result;
      if (!stan::prob::check_size_match(function,y.size(),Sigma.cols(),&result, Policy()))
	return result;
      if (!stan::prob::check_x(function, y, &result, Policy())) 
	return result;
      if (!stan::prob::check_cov_matrix(function, Sigma, &result, Policy())) 
	return result;
      
      
      typename promote_args<T_y,T_loc,T_covar>::type lp(0.0);
      if (y.rows() == 0)
	return lp;
      if (!propto) 
	lp += NEG_LOG_SQRT_TWO_PI * y.rows();
      if (!propto || !is_constant<T_covar>::value)
	lp -= 0.5 * log(Sigma.determinant());	
      if (!propto 
	  || !is_constant<T_y>::value
	  || !is_constant<T_loc>::value
	  || !is_constant<T_covar>::value)
	lp -= 0.5 * ((y - mu).transpose() * Sigma.inverse() * (y - mu))(0,0);
      return lp;
    }

    // MultiNormal(y|mu,L)       [y.rows() = mu.rows() = L.rows() = L.cols();
    //                            y.cols() = mu.cols() = 0;
    //                            Sigma = LL' with L a Cholesky factor]
    /**
     * The log of the multivariate normal density for the given y, mu, and
     * L (a Cholesky factor of Sigma, a variance matrix).
     * Sigma = LL', a square, semi-positive definite matrix.
     * Dimension, d, is implicit.
     *
     * \f{eqnarray*}{
     y &\sim& \mbox{\sf{N}} (\mu, LL') \\
     \log (p (y \,|\, \mu, L) ) &=& \log \left( (2 \pi)^{-d/2} \left| LL' \right|^{-1/2} \times \exp \left(-\frac{1}{2}(y - \mu)^T (LL')^{-1} (y - \mu) \right) \right)
     \f}
     * 
     * 
     * @param y A scalar vector
     * @param mu The mean vector of the multivariate normal distribution.
     * @param L The Cholesky decomposition of a variance matrix of the multivariate normal distribution
     * @return The log of the multivariate normal density.
     * @throw std::domain_error if LL' is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_loc Type of location.
     * @tparam T_covar Type of scale.
     */
    template <bool propto = false, 
	      typename T_y, typename T_loc, typename T_covar, 
	      class Policy = policy<> > 
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(const Matrix<T_y,Dynamic,1>& y,
		     const Matrix<T_loc,Dynamic,1>& mu,
		     const Eigen::TriangularView<T_covar,Eigen::Lower>& L,
		     const Policy& = Policy()) {
      static const char* function = "stan::prob::multi_normal_log<%1%>(%1%)";
      
      typename promote_args<T_y,T_loc,T_covar>::type lp(0.0);
      if (!stan::prob::check_x(function, y, &lp, Policy())) 
	return lp;
      // FIXME: checks on L
      
      if (y.rows() == 0)
	return lp;

      if (!propto) 
	lp += NEG_LOG_SQRT_TWO_PI * y.rows();
      if (!propto
	  || !is_constant<T_covar>::value)
	lp -= log(L.diagonal().array().prod());

      if (!propto 
	  || !is_constant<T_y>::value 
	  || !is_constant<T_loc>::value
	  || !is_constant<T_covar>::value) {
	Matrix<T_covar,Dynamic,1> half = L.solveTriangular(Matrix<T_covar,Dynamic,Dynamic>(L.rows(),L.rows()).setOnes()) * (y - mu);
	lp -= 0.5 * half.dot(half);
      }
      return lp;
    }
     
  }
}

#endif
