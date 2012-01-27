#ifndef __STAN__PROB__DISTRIBUTIONS__MULTI_NORMAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTI_NORMAL_HPP__

#include <stan/maths/matrix.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    // MultiNormal(y|mu,Sigma)   [y.rows() = mu.rows() = Sigma.rows();
    //                            y.cols() = mu.cols() = 0;
    //                            Sigma symmetric, non-negative, definite]
    /**
     * The log of the multivariate normal density for the given y, mu, and
     * variance matrix. 
     * The variance matrix, Sigma, must be size d x d, symmetric,
     * and semi-positive definite. Dimension, d, is implicit.
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
              class Policy = boost::math::policies::policy<> > 
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                     const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                     const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                     const Policy& = Policy()) {
      static const char* function = "stan::prob::multi_normal_log<%1%>(%1%)";

      typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type 
        lp(0.0);

      if (!check_size_match(function, y.size(), mu.size(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, y.size(), Sigma.rows(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, y.size(), Sigma.cols(), &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter, mu", &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "y", &lp, Policy())) 
        return lp;
      if (!check_cov_matrix(function, Sigma, &lp, Policy()))
        return lp;
      
      using stan::maths::multiply_log;
      using stan::maths::subtract;
      using stan::maths::determinant;
      using stan::maths::inverse;
      using stan::maths::multiply;
      using stan::maths::transpose;

      if (y.rows() == 0)
        return lp;
      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * y.rows();
      if (include_summand<propto,T_covar>::value)
        lp -= multiply_log(0.5,determinant(Sigma));
      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        Eigen::Matrix<typename boost::math::tools::promote_args<T_y,T_loc>::type, Eigen::Dynamic, 1> diff 
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
              class Policy = boost::math::policies::policy<> > 
    inline typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type
    multi_normal_cholesky_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                              const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                              const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L,
                              const Policy& = Policy()) {
      static const char* function = "stan::prob::multi_normal_log<%1%>(%1%)";

      using stan::maths::multiply;
      using stan::maths::subtract;
      
      typename boost::math::tools::promote_args<T_y,T_loc,T_covar>::type lp(0.0);

      if (!check_size_match(function, y.size(), mu.size(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, y.size(), L.rows(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, y.size(), L.cols(), &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "y", &lp, Policy())) 
        return lp;

      if (y.rows() == 0)
        return lp;

      if (include_summand<propto>::value) 
        lp += NEG_LOG_SQRT_TWO_PI * y.rows();

      if (include_summand<propto,T_covar>::value)
        lp -= L.diagonal().array().log().sum();

      if (include_summand<propto,T_y,T_loc,T_covar>::value) {
        
        Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic> 
          L_inv(L
                .template triangularView<Eigen::Lower>()
                .solve(Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>
                       ::Identity(L.rows(),L.rows())));
        
        Eigen::Matrix<typename boost::math::tools::promote_args<T_covar,T_loc,T_y>::type, Eigen::Dynamic, 1> 
          half(multiply(L_inv,
                        subtract(y,mu)));

        lp -= 0.5 * half.dot(half);  // FIXME:  add dot_self function + deriv, fold in half

      }
      return lp;
    }
     
  }
}

#endif
