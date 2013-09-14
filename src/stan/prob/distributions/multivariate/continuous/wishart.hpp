#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__WISHART_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__WISHART_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/agrad/matrix.hpp>
#include <stan/prob/traits.hpp>
#include <boost/concept_check.hpp>
#include "stan/prob/distributions/univariate/continuous/normal.hpp"
#include "stan/prob/distributions/univariate/continuous/chi_square.hpp"
#include <stan/math/functions/lmgamma.hpp>
#include <stan/math/matrix/columns_dot_product.hpp>
#include <stan/math/matrix/trace.hpp>
#include <stan/math/matrix/ldlt.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>

namespace stan {

  namespace prob {

    // Wishart(Sigma|n,Omega)  [Sigma, Omega symmetric, non-neg, definite; 
    //                          Sigma.dims() = Omega.dims();
    //                           n > Sigma.rows() - 1]
    /**
     * The log of the Wishart density for the given W, degrees of freedom, 
     * and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * nu must be greater than k-1
     *
     * \f{eqnarray*}{
     W &\sim& \mbox{\sf{Wishart}}_{\nu} (S) \\
     \log (p (W \,|\, \nu, S) ) &=& \log \left( \left(2^{\nu k/2} \pi^{k (k-1) /4} \prod_{i=1}^k{\Gamma (\frac{\nu + 1 - i}{2})} \right)^{-1} 
     \times \left| S \right|^{-\nu/2} \left| W \right|^{(\nu - k - 1) / 2}
     \times \exp (-\frac{1}{2} \mbox{tr} (S^{-1} W)) \right) \\
     &=& -\frac{\nu k}{2}\log(2) - \frac{k (k-1)}{4} \log(\pi) - \sum_{i=1}^{k}{\log (\Gamma (\frac{\nu+1-i}{2}))}
     -\frac{\nu}{2} \log(\det(S)) + \frac{\nu-k-1}{2}\log (\det(W)) - \frac{1}{2} \mbox{tr} (S^{-1}W)
     \f}
     * 
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <bool propto,
              typename T_y, typename T_dof, typename T_scale>
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
                const T_dof& nu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S) {
      static const char* function = "stan::prob::wishart_log(%1%)";

      using stan::math::check_greater;
      using stan::math::check_size_match;
      using boost::math::tools::promote_args;

      typename Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>::size_type k = W.rows();
      typename promote_args<T_y,T_dof,T_scale>::type lp(0.0);
      if (!check_greater(function, nu, k-1, 
                         "Degrees of freedom parameter", &lp))
        return lp;
      if (!check_size_match(function, 
                            W.rows(), "Rows of random variable",
                            W.cols(), "columns of random variable",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            S.rows(), "Rows of scale parameter",
                            S.cols(), "columns of scale parameter",
                            &lp))
        return lp;
      if (!check_size_match(function, 
                            W.rows(), "Rows of random variable",
                            S.rows(), "columns of scale parameter",
                            &lp))
        return lp;
      // FIXME: domain checks

      using stan::math::log_determinant_ldlt;
      using stan::math::mdivide_left_ldlt;
      using stan::math::LDLT_factor;
      
      LDLT_factor<T_y,Eigen::Dynamic,Eigen::Dynamic> ldlt_W(W);
      if (!ldlt_W.success()) {
        std::ostringstream message;
        message << "W is not positive definite (%1%).";
        std::string str(message.str());
        stan::math::dom_err(function,W(0,0),"W",str.c_str(),"",&lp);
        return lp;
      }
      LDLT_factor<T_scale,Eigen::Dynamic,Eigen::Dynamic> ldlt_S(S);
      if (!ldlt_S.success()) {
        std::ostringstream message;
        message << "S is not positive definite (%1%).";
        std::string str(message.str());
        stan::math::dom_err(function,S(0,0),"S",str.c_str(),"",&lp);
        return lp;
      }
      
      using stan::math::trace;
      using stan::math::lmgamma;
      if (include_summand<propto,T_dof>::value)
        lp += nu * k * NEG_LOG_TWO_OVER_TWO;

      if (include_summand<propto,T_dof>::value)
        lp -= lmgamma(k, 0.5 * nu);

      if (include_summand<propto,T_dof,T_scale>::value)
        lp -= 0.5 * nu * log_determinant_ldlt(ldlt_S);

      if (include_summand<propto,T_scale,T_y>::value) {
//        L_S = crossprod(mdivide_left_tri_low(L_S));
//        Eigen::Matrix<T_scale,Eigen::Dynamic,1> S_inv_vec = Eigen::Map<
//          const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic> >(
//                                                                       &L_S(0), L_S.size(), 1);
//        Eigen::Matrix<T_y,Eigen::Dynamic,1> W_vec = Eigen::Map<
//          const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> >(
//                                                                   &W(0), W.size(), 1);
//        lp -= 0.5 * dot_product(S_inv_vec, W_vec); // trace(S^-1 * W)
        Eigen::Matrix<typename promote_args<T_y,T_scale>::type,Eigen::Dynamic,Eigen::Dynamic> Sinv_W(mdivide_left_ldlt(ldlt_S,W));
        lp -= 0.5*trace(Sinv_W);
      }

      if (include_summand<propto,T_y,T_dof>::value && nu != (k + 1))
        lp += 0.5*(nu - k - 1.0) * log_determinant_ldlt(ldlt_W);
      return lp;
    }

    template <typename T_y, typename T_dof, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
                const T_dof& nu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S) {
      return wishart_log<false>(W,nu,S);
    }

    template <class RNG>
    inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
    wishart_rng(const double nu,
                const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& S,
                RNG& rng) {

      static const char* function = "stan::prob::wishart_rng(%1%)";

      using stan::math::check_size_match;
      using stan::math::check_positive;

      check_positive(function,nu,"degrees of freedom");
      check_size_match(function, 
                       S.rows(), "Rows of scale parameter",
                       S.cols(), "columns of scale parameter");

      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> B(S.rows(), S.cols());
      B.setZero();

      for(int i = 0; i < S.cols(); i++) {
        B(i,i) = std::sqrt(chi_square_rng(nu - i, rng));
        for(int j = 0; j < i; j++)
          B(j,i) = normal_rng(0,1,rng);
      }

      return stan::math::multiply_lower_tri_self_transpose(S.llt().matrixL() * B);
    }
  }
}
#endif
