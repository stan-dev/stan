#ifndef __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__WISHART_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__WISHART_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/traits.hpp>
#include <boost/concept_check.hpp>

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
              typename T_y, typename T_dof, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
                const T_dof& nu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S,
                const Policy&) {
      static const char* function = "stan::prob::wishart_log<%1%>(%1%)";

      using stan::math::check_greater_or_equal;
      using stan::math::check_size_match;
      using boost::math::tools::promote_args;
      using Eigen::Array;

      unsigned int k = W.rows();
      typename promote_args<T_y,T_dof,T_scale>::type lp;
      if (!check_greater_or_equal(function, nu, k - 1, 
                                  "Degrees of freedom, nu,", &lp, Policy()))
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
      Eigen::LLT< Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic> > LLT_S = S.llt();
      if (LLT_S.info() != Eigen::Success) {
        lp = stan::math::policies::raise_domain_error<T_scale>(function,
                                              "S is not positive definite (%1%)",
                                              0,Policy());
        return lp;
      }      

      Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> L_W = LLT_W.matrixL();
      Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic> L_S = LLT_S.matrixL();

      using stan::math::multiply;
//      using stan::math::elt_multiply;
      using stan::math::mdivide_left_tri;
      using stan::math::lmgamma;
      lp = 0.0;
      if (include_summand<propto,T_dof>::value)
        lp += nu * k * NEG_LOG_TWO_OVER_TWO;

      if (include_summand<propto,T_dof>::value)
        lp -= lmgamma(k, 0.5 * nu);

      if (include_summand<propto,T_dof,T_scale>::value)
	lp -= nu * L_S.diagonal().array().log().sum();

      if (include_summand<propto,T_scale,T_y>::value) {
	Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic> I(k,k);
	I.setIdentity();
	L_S = mdivide_left_tri<Eigen::Lower>(L_S, I);
	L_S = L_S.transpose() * L_S.template triangularView<Eigen::Lower>();
	lp -= 0.5 * multiply(L_S, W).diagonal().array().sum();	
// 	lp -= 0.5 * elt_multiply(L_S, S).array().sum(); // FIXME: this way would be better but does not build
      }

      if (include_summand<propto,T_y,T_dof>::value && nu != (k + 1))
	lp += (nu - k - 1.0) * L_W.diagonal().array().log().sum();

      return lp;
    }

    template <bool propto,
              typename T_y, typename T_dof, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
                const T_dof& nu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S) {
      return wishart_log<propto>(W,nu,S,stan::math::default_policy());
    }


    template <typename T_y, typename T_dof, typename T_scale, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
                const T_dof& nu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S,
                const Policy&) {
      return wishart_log<false>(W,nu,S,Policy());
    }


    template <typename T_y, typename T_dof, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    wishart_log(const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& W,
                const T_dof& nu,
                const Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic>& S) {
      return wishart_log<false>(W,nu,S,stan::math::default_policy());
    }


  }
}
#endif
