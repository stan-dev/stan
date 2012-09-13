#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__INV_WISHART_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__INV_WISHART_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {
    // InvWishart(Sigma|n,Omega)  [W, S symmetric, non-neg, definite; 
    //                             W.dims() = S.dims();
    //                             n > S.rows() - 1]
    /**
     * The log of the Inverse-Wishart density for the given W, degrees
     * of freedom, and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive 
     * definite.
     *
     * \f{eqnarray*}{
     W &\sim& \mbox{\sf{Inv-Wishart}}_{\nu} (S) \\
     \log (p (W \,|\, \nu, S) ) &=& \log \left( \left(2^{\nu k/2} \pi^{k (k-1) /4} \prod_{i=1}^k{\Gamma (\frac{\nu + 1 - i}{2})} \right)^{-1} 
     \times \left| S \right|^{\nu/2} \left| W \right|^{-(\nu + k + 1) / 2}
     \times \exp (-\frac{1}{2} \mbox{tr} (S W^{-1})) \right) \\
     &=& -\frac{\nu k}{2}\log(2) - \frac{k (k-1)}{4} \log(\pi) - \sum_{i=1}^{k}{\log (\Gamma (\frac{\nu+1-i}{2}))}
     +\frac{\nu}{2} \log(\det(S)) - \frac{\nu+k+1}{2}\log (\det(W)) - \frac{1}{2} \mbox{tr}(S W^{-1})
     \f}
     * 
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Inverse-Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not 
     * semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <bool propto,
              typename T_y, typename T_dof, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
                const T_dof& nu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S,
                const Policy&) {
      static const char* function = "stan::prob::inv_wishart_log(%1%)";
      
      using stan::math::check_greater_or_equal;
      using stan::math::check_size_match;
      using boost::math::tools::promote_args;
      using Eigen::Array;

      typename Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>::size_type k = S.rows();
      typename promote_args<T_y,T_dof,T_scale>::type lp(0.0);
      if(!check_greater_or_equal(function, nu, k-1, "Degrees of freedom parameter", 
                                 &lp, Policy()))
        return lp;
      if (!check_size_match(function, W.rows(), W.cols(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, S.rows(), S.cols(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, W.rows(), S.rows(), &lp, Policy()))
        return lp;
      // FIXME: domain checks
        
      Eigen::LLT< Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> > LLT_W = W.llt();
      if (LLT_W.info() != Eigen::Success) {
        lp = stan::math::policies::raise_domain_error<T_y>(function,
                                              "W is not positive definite (%1%)",
                                              0,Policy());
        return lp;
      }
      Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> L = LLT_W.matrixL();

      using stan::math::elt_multiply;
      using stan::math::mdivide_left_tri_low;
      using stan::math::lmgamma;
      
      if (include_summand<propto,T_dof>::value)
        lp -= lmgamma(k, 0.5 * nu);
      if (include_summand<propto,T_dof,T_scale>::value) {
        lp += nu * S.llt().matrixLLT().diagonal().array().log().sum();
      }
      if (include_summand<propto,T_y,T_dof,T_scale>::value) {
        lp -= (nu + k + 1.0) * L.diagonal().array().log().sum();
      }
      if (include_summand<propto,T_y,T_scale>::value) {
        Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> I(k,k);
        I.setIdentity();
        L = mdivide_left_tri_low(L, I);
        L = L.transpose() * L.template triangularView<Eigen::Lower>();
        lp -= 0.5 * elt_multiply(S, L).array().sum(); // trace(S * W^-1)
      }
      if (include_summand<propto,T_dof,T_scale>::value)
        lp += nu * k * NEG_LOG_TWO_OVER_TWO;
      return lp;
    }

    template <bool propto,
              typename T_y, typename T_dof, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
              const T_dof& nu,
              const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S) {
      return inv_wishart_log<propto>(W,nu,S,stan::math::default_policy());
    }


    template <typename T_y, typename T_dof, typename T_scale, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
                const T_dof& nu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S,
                const Policy&) {
      return inv_wishart_log<false>(W,nu,S,Policy());
    }


    template <typename T_y, typename T_dof, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
            const T_dof& nu,
            const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S) {
      return inv_wishart_log<false>(W,nu,S,stan::math::default_policy());
    }


  }
}
#endif
