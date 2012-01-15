#ifndef __STAN__PROB__DISTRIBUTIONS__LKJ_CORR_HPP__
#define __STAN__PROB__DISTRIBUTIONS__LKJ_CORR_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/distributions/beta.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    using Eigen::Matrix;
    using Eigen::Dynamic;

    template <typename T_shape>
    T_shape do_ljk_constant(const T_shape& eta, const unsigned int& K) {
       // Lewandowski, Kurowicka, and Joe (2009) equations 15 and 16
      T_shape the_sum = 0.0;
      T_shape constant = 0.0;
      T_shape beta_arg;
      
      if(eta == 1.0) {
	for(unsigned int k = 1; k < K; k++) { // yes, go from 1 to K - 1
	  beta_arg = 0.5 * (k + 1.0);
	  constant += k * beta_log<true>(beta_arg, beta_arg);
	  the_sum += pow(static_cast<double>(k),2.0);
	}
	constant += the_sum * LOG_TWO;
	return constant;
      }

      T_shape diff;
      for(unsigned int k = 1; k < K; k++) { // yes, go from 1 to K - 1
	diff = K - k;
	beta_arg = eta + 0.5 * (diff - 1);
	constant += diff * beta_log<true>(beta_arg, beta_arg);
	the_sum += (2.0 * eta - 2.0 + diff) * diff;
      }
      constant += the_sum * LOG_TWO;
      return constant;
    }

    // LKJ_Corr(y|eta) [ y correlation matrix (not covariance matrix)
    //                  eta > 0; eta == 1 <-> uniform]
    template <bool propto = false,
	      typename T_y, typename T_shape, 
	      class Policy = policy<> >
    inline typename promote_args<T_y, T_shape>::type
    lkj_corr_log(const Matrix<T_y,Dynamic,Dynamic>& y, const T_shape& eta, const Policy& /* pol */) {
      static const char* function = "stan::prob::multi_normal_log<%1%>(%1%)";

      if (!check_size_match(function, y.rows(), mu.size(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, y.rows(), y.cols(), &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "y", &lp, Policy())) 
        return lp;

      typename promote_args<T_y,T_shape>::type lp(0.0);
      
      const unsigned int K = y.rows();
      if (K == 0)
        return lp;

      LLT<T_y, Dynamic, Dynamic> Cholesky = y.llt();
      if(Cholesky.info() == NumericalIssue) // do we need a check_numerical_issue function?
	return lp;
      
      lp = lkj_corr_cholesky_log<propto>(Cholesky.matrixL(), eta);
      return lp;
    }

    // LKJ_Corr(L|eta) [ L Cholesky factor of correlation matrix
    //                  eta > 0; eta == 1 <-> uniform]
    template <bool propto = false,
	      typename T_covar, typename T_shape, 
	      class Policy = policy<> >
    inline typename promote_args<T_covar, T_shape>::type
    lkj_corr_cholesky_log(const Matrix<T_covar,Dynamic,Dynamic>& L, 
			  const T_shape& eta, const Policy& /* pol */) {

      typename promote_args<T_covar,T_shape>::type lp(0.0);
      const unsigned int K = L.rows();
      if (K == 0)
        return lp;

      if (include_summand<propto>::value) 
        lp += do_ljk_constant(eta, K);
      if (eta != 1.0 && include_summand<propto,T_covar,T_shape>::value)
        lp += (eta - 1.0) * 2.0 * L.diagonal().array().log().sum();
      
      return lp;
    }

  }
}
#endif
