#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_SUFFICIENT_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__MULTI_NORMAL_SUFFICIENT_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/error_handling/matrix/check_ldlt_factor.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/math/matrix/trace_inv_quad_form_ldlt.hpp>
#include <stan/math/matrix/log_determinant_ldlt.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    template <typename T_sample, typename T_loc, typename T_covar>
    typename boost::math::tools::promote_args<T_sample, typename scalar_type<T_loc>::type, T_covar>::type
    multi_normal_sufficient_log(const int sampleSize,
				const Eigen::Matrix<T_sample,Eigen::Dynamic,1>& sampleMu,
				const Eigen::Matrix<T_sample,Eigen::Dynamic,Eigen::Dynamic>& sampleSigma,
				const T_loc& mu,
				const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      static const std::string function("stan::prob::multi_normal_sufficient_log");
      typedef typename boost::math::tools::promote_args<T_sample, typename scalar_type<T_loc>::type, T_covar>::type param_t;
      typedef param_t lp_type;
      lp_type lp(0.0);
	   
      using stan::error_handling::check_size_match;
      using stan::error_handling::check_finite;
      using stan::error_handling::check_not_nan;
      using stan::error_handling::check_positive;
      using stan::error_handling::check_symmetric;
      using stan::error_handling::check_ldlt_factor;
	   
      check_size_match(function,
                       "Rows of covariance parameter", sampleSigma.rows(), 
                       "columns of covariance parameter", sampleSigma.cols());
      check_positive(function, "Covariance matrix rows", sampleSigma.rows());
      check_symmetric(function, "Covariance matrix", sampleSigma);

      check_size_match(function,
                       "Rows of covariance parameter", Sigma.rows(), 
                       "columns of covariance parameter", Sigma.cols());
      check_positive(function, "Covariance matrix rows", Sigma.rows());
      check_symmetric(function, "Covariance matrix", Sigma);
      
      check_size_match(function, 
                       "Size of data location", sampleMu.size(),
                       "size of model location", mu.size());
      check_size_match(function, 
                       "Size of data covariance", sampleSigma.rows(), 
                       "size of model covariance", Sigma.rows());
  
      stan::math::LDLT_factor<param_t,Eigen::Dynamic,Eigen::Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function, "LDLT_Factor of covariance parameter", ldlt_Sigma);

      Eigen::Matrix<param_t, Eigen::Dynamic, Eigen::Dynamic> ss;
      ss = mdivide_left_ldlt(ldlt_Sigma, sampleSigma);

      lp += (ss.diagonal().sum() + log_determinant_ldlt(ldlt_Sigma)) * (sampleSize - 1);

      lp_type lp_location(0.0);
      {
	Eigen::Matrix<param_t, Eigen::Dynamic, 1> y_minus_mu(mu.size());

	for (int j = 0; j < mu.size(); j++)
	  y_minus_mu(j) = mu(j) - sampleMu(j);

	lp_location = trace_inv_quad_form_ldlt(ldlt_Sigma, y_minus_mu) * sampleSize;
	// Could avoid re-solving Sigma
	// lp_location = quad_form(ss, y_minus_mu).diagonal().sum() * sampleSize;
      }
      return (lp + lp_location) * -0.5;
    }
  }
}

#endif
