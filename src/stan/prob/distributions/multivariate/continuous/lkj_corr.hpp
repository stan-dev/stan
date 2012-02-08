#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_CORR_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__LKJ_CORR_HPP__

#include <stan/prob/constants.hpp>
#include <stan/maths/matrix_error_handling.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    using Eigen::Matrix;
    using Eigen::Dynamic;
    using Eigen::LLT;
    using Eigen::NumericalIssue;    
    
    template <typename T_shape>
    T_shape do_lkj_constant(const T_shape& eta, const unsigned int& K) {
      // Lewandowski, Kurowicka, and Joe (2009) equations 15 and 16
      T_shape the_sum = 0.0;
      T_shape constant = 0.0;
      T_shape beta_arg;
      
      if(eta == 1.0) {
        for(unsigned int k = 1; k < K; k++) { // yes, go from 1 to K - 1
          beta_arg = 0.5 * (k + 1.0);
          constant += k * (2.0 * lgamma(beta_arg) - lgamma(2.0 * beta_arg));
          the_sum += pow(static_cast<double>(k),2.0);
        }
        constant += the_sum * LOG_TWO;
        return constant;
      }

      T_shape diff;
      for(unsigned int k = 1; k < K; k++) { // yes, go from 1 to K - 1
        diff = K - k;
        beta_arg = eta + 0.5 * (diff - 1);
        constant += diff * (2.0 * lgamma(beta_arg) - lgamma(2.0 * beta_arg));
        the_sum += (2.0 * eta - 2.0 + diff) * diff;
      }
      constant += the_sum * LOG_TWO;
      return constant;
    }

    // LKJ_Corr(L|eta) [ L Cholesky factor of correlation matrix
    //                  eta > 0; eta == 1 <-> uniform]
    template <bool propto = false,
              typename T_covar, typename T_shape, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_covar, T_shape>::type
    lkj_corr_cholesky_log(const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L, 
                          const T_shape& eta, 
			  const Policy& = Policy() ) {
      static const char* function = "stan::prob::lkj_corr_cholesky_log<%1%>(%1%)";

      using boost::math::tools::promote_args;
      using stan::maths::check_positive;
      
      typename promote_args<T_covar,T_shape>::type lp(0.0);
      if (!check_positive(function, eta, "eta", &lp, Policy()))
	return lp;      

      const unsigned int K = L.rows();
      if (K == 0)
        return 0.0;
      
      if (include_summand<propto>::value) 
        lp += do_lkj_constant(eta, K);
      if (include_summand<propto,T_covar,T_shape>::value && eta != 1.0)
        lp += (eta - 1.0) * 2.0 * L.diagonal().array().log().sum();
      
      return lp;
    }


    // LKJ_Corr(y|eta) [ y correlation matrix (not covariance matrix)
    //                  eta > 0; eta == 1 <-> uniform]
    template <bool propto = false,
              typename T_y, typename T_shape, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_y, T_shape>::type
    lkj_corr_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y, 
		 const T_shape& eta, 
		 const Policy& = Policy() ) {
      static const char* function = "stan::prob::lkj_corr_log<%1%>(%1%)";

      using stan::maths::check_size_match;
      using stan::maths::check_not_nan;
      using stan::maths::check_positive;
      using stan::maths::check_corr_matrix;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_shape>::type lp;
      if (!check_positive(function, eta, "eta", &lp, Policy()))
	return lp;      
      if (!check_size_match(function, y.rows(), y.cols(), &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "y", &lp, Policy())) 
        return lp;
      if (!check_corr_matrix(function, y, &lp, Policy())) {
	return lp;
      }
      
      const unsigned int K = y.rows();
      if (K == 0)
        return 0.0;

      LLT< Matrix<T_y, Dynamic, Dynamic> > Cholesky = y.llt();
      if(Cholesky.info() == Eigen::NumericalIssue) // do we need a check_numerical_issue function?
        return lp;

      Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> L = Cholesky.matrixL();
      return lkj_corr_cholesky_log<propto>(L, eta, Policy());
    }


  }
}
#endif
