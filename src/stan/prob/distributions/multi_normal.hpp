#ifndef __STAN__PROB__DISTRIBUTIONS__MULTI_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTI_NORMAL_HPP__

#include <stan/prob/distributions_error_handling.hpp>
#include <stan/prob/distributions_constants.hpp>
#include <stan/prob/traits.hpp>

#include <stan/maths/matrix.hpp>



namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    using stan::maths::determinant;
    using stan::maths::dot_product;
    using stan::maths::inverse;
    using stan::maths::multiply;
    using stan::maths::transpose;

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
     \log (p (y \,|\, \mu, \Sigma) ) 
     &=& \log \left( (2 \pi)^{-d/2} \left| \Sigma \right|^{-1/2} 
         \times 
         \exp \left(-\frac{1}{2}(y - \mu)^T \Sigma^{-1} 
                                (y - \mu) \right) \right) \\
     &=& -\frac{d}{2}\log (2 \pi) 
         - \frac{1}{2} \log (\det(\Sigma)) 
         - \frac{1}{2} (y - \mu)^T \Sigma^{-1} (y - \mu)
     \f}
     * 
     * 
     * @param y A scalar vector
     * @param mu The mean vector of the multivariate normal distribution.
     * @param Sigma The variance matrix of the multivariate normal distribution
     * @return The log of the multivariate normal density.
     * @throw std::domain_error if Sigma is not square, not symmetric, 
     * or not semi-positive definite.
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
      
      typename promote_args<T_y,T_loc,T_covar>::type lp(0.0);
      if (!check_size_match(function, y.size(), mu.size(), &lp, Policy()))
	return lp;
      if (!check_size_match(function, y.size(), Sigma.rows(), &lp, Policy()))
	return lp;
      if (!check_size_match(function, y.size(), Sigma.cols(), &lp, Policy()))
	return lp;
      if (!check_x(function, y, &lp, Policy())) 
	return lp;
      if (!check_cov_matrix(function, Sigma, &lp, Policy()))
	return lp;
      if (y.rows() == 0)
	return lp;
      if (include_summand<propto>::value) 
	lp += NEG_LOG_SQRT_TWO_PI * y.rows();
      if (include_summand<propto,T_covar>::value)
	lp -= 0.5 * log(determinant(Sigma));
      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
	Eigen::Matrix<typename promote_args<T_y,T_loc>::type, Dynamic, 1> diff 
	  = subtract(y,mu);
	lp -= 0.5 * multiply(multiply(transpose(diff),inverse(Sigma)),
			     diff);
      }
      return lp;
    }

    /**
     * The log of the multivariate normal density for the given y, mu, and
     * a Cholesky factor L of the variance matrix.
     * Sigma = LL', a square, semi-positive definite matrix.
     *
     * \f${eqnarray*}
     * y &\sim& \mbox{\sf{N}} (\mu, LL') \\
     * \log (p (y \,|\, \mu, L) )
     * &=& \log \left( (2 \pi)^{-d/2} \left| LL' \right|^{-1/2} 
     * \times \exp \left(-\frac{1}{2}(y - \mu)^T (LL')^{-1} (y - \mu) 
     * \right) \right)
     \f$
     * 
     * @param y A scalar vector
     * @param mu The mean vector of the multivariate normal distribution.
     * @param L The Cholesky decomposition of a variance matrix 
     * of the multivariate normal distribution
     * @return The log of the multivariate normal density.
     * @throw std::domain_error if LL' is not square, not symmetric, 
     * or not semi-positive definite.
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

      if (!check_size_match(function, y.size(), mu.size(), &lp, Policy()))
	return lp;
      if (!check_size_match(function, y.size(), L.rows(), &lp, Policy()))
	return lp;
      if (!check_size_match(function, y.size(), L.cols(), &lp, Policy()))
	return lp; // redundant with Triangular View?
      if (!check_x(function, y, &lp, Policy())) 
	return lp;

      // FIXME: checks on L -- what are they?
      
      if (y.rows() == 0)
	return lp;

      if (include_summand<propto>::value) 
	lp += NEG_LOG_SQRT_TWO_PI * y.rows();
      if (include_summand<propto,T_covar>::value) 
	lp -= log(L.diagonal().array().prod());
      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
	Matrix<T_covar,Dynamic,1> half 
	  = L.solveTriangular(Matrix<T_covar,Dynamic,Dynamic>(L.rows(),L.rows())
			      .setOnes())
	  * (y - mu);
	lp -= 0.5 * dot_product(half,half);
      }
      return lp;
    }
     
  }
}

#endif
