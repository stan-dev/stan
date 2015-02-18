#ifndef STAN__MATH__PRIM__MAT__PROB__LKJ_CORR_CHOLESKY_LOG_HPP
#define STAN__MATH__PRIM__MAT__PROB__LKJ_CORR_CHOLESKY_LOG_HPP

#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/beta.hpp>
#include <stan/math/prim/scal/meta/prob_traits.hpp>
#include <stan/math/prim/scal/fun/transform.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_log.hpp>

namespace stan {
  namespace prob {

    // LKJ_Corr(L|eta) [ L Cholesky factor of correlation matrix
    //                  eta > 0; eta == 1 <-> uniform]
    template <bool propto,
              typename T_covar, typename T_shape>
    typename boost::math::tools::promote_args<T_covar, T_shape>::type
    lkj_corr_cholesky_log(
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L, 
                          const T_shape& eta) {

      static const char* function("stan::prob::lkj_corr_cholesky_log");

      using boost::math::tools::promote_args;
      using stan::math::check_positive;
      using stan::math::check_lower_triangular;
      using stan::math::sum;
      
      typename promote_args<T_covar,T_shape>::type lp(0.0);
      check_positive(function, "Shape parameter", eta);
      check_lower_triangular(function, "Random variable", L);

      const unsigned int K = L.rows();
      if (K == 0)
        return 0.0;
            
      if (include_summand<propto,T_shape>::value) 
        lp += do_lkj_constant(eta, K);
      if (include_summand<propto,T_covar,T_shape>::value) {
        const int Km1 = K - 1;
        Eigen::Matrix<T_covar,Eigen::Dynamic,1> log_diagonals =
          L.diagonal().tail(Km1).array().log();
        Eigen::Matrix<T_covar,Eigen::Dynamic,1> values(Km1);
        for (int k = 0; k < Km1; k++)
          values(k) = (Km1 - k - 1) * log_diagonals(k);
        if ( (eta == 1.0) &&
             stan::is_constant<typename stan::scalar_type<T_shape> >::value) {
          lp += sum(values);
          return(lp);
        }
        values += (2.0 * eta - 2.0) * log_diagonals;
        lp += sum(values);
      }
      
      return lp;
    }

    template <typename T_covar, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_covar, T_shape>::type
    lkj_corr_cholesky_log(
                          const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& L, 
                          const T_shape& eta) {
      return lkj_corr_cholesky_log<false>(L,eta);
    }

  }
}
#endif
