#ifndef __STAN__PROB__DISTRIBUTIONS_INV_WISHART_HPP__
#define __STAN__PROB__DISTRIBUTIONS_INV_WISHART_HPP__

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/policies/policy.hpp>

#include "stan/prob/transform.hpp"
#include "stan/prob/distributions_error_handling.hpp"
#include "stan/prob/distributions_constants.hpp"

namespace stan {
  namespace prob {
    using namespace std;
    using namespace stan::maths;
    using Eigen::Dynamic;
    using Eigen::Matrix;

    // InvWishart(Sigma|n,Omega)  [W, S symmetric, non-neg, definite; 
    //                             W.dims() = S.dims();
    //                             n > S.rows() - 1]
    /**
     * The log of the Inverse-Wishart density for the given W, degrees of freedom, 
     * and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * nu must be greater than k-1
     *
     * \f{eqnarray*}{
       W &\sim& \mbox{\sf{Inv-Wishart}}_{\nu} (S) \\
       \log (p (W \,|\, \nu, S) ) &=& \log \left( \left(2^{\nu k/2} \pi^{k (k-1) /4} \prod_{i=1}^k{\Gamma (\frac{\nu + 1 - i}{2})} \right)^{-1} 
                                                  \times \left| S \right|^{\nu/2} \left| W \right|^{-(\nu + k + 1) / 2}
						  \times \exp (-\frac{1}{2} \mathsf{tr} (S W^{-1})) \right) \\
       &=& -\frac{\nu k}{2}\log(2) - \frac{k (k-1)}{4} \log(\pi) - \sum_{i=1}^{k}{\log (\Gamma (\frac{\nu+1-i}{2}))}
           +\frac{\nu}{2} \log(\det(S)) - \frac{\nu+k+1}{2}\log (\det(W)) - \frac{1}{2} \mathsf{tr}(S W^{-1})
     \f}
     * 
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Inverse-Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_dof, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_log(const Matrix<T_y,Dynamic,Dynamic>& W,
		    const T_dof& nu,
		    const Matrix<T_scale,Dynamic,Dynamic>& S,
		    const Policy& /* pol */) {
      static const char* function = "stan::prob::wishart_log<%1%>(%1%)";

      unsigned int k = S.rows();
      double result;
      if(!stan::prob::check_positive(function, nu - (k-1), "Degrees of freedom - k-1", &result, Policy()))
	return result;
      // FIXME: domain checks

      return 0.5 * nu * log(S.determinant())
	- 0.5 * (nu + k + 1.0) * log(W.determinant())
	- 0.5 * (S * W.inverse()).trace()
	+  nu * k * NEG_LOG_TWO_OVER_TWO
	- lmgamma(k, 0.5 * nu);
    }

        /**
     * The log of the Inverse-Wishart density for the given W, degrees of freedom, 
     * and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * nu must be greater than k-1
     *
     * \f{eqnarray*}{
       W &\sim& \mbox{\sf{Inv-Wishart}}_{\nu} (S) \\
       \log (p (W \,|\, \nu, S) ) &=& \log \left( \left(2^{\nu k/2} \pi^{k (k-1) /4} \prod_{i=1}^k{\Gamma (\frac{\nu + 1 - i}{2})} \right)^{-1} 
                                                  \times \left| S \right|^{\nu/2} \left| W \right|^{-(\nu + k + 1) / 2}
						  \times \exp (-\frac{1}{2} \mathsf{tr} (S W^{-1})) \right) \\
       &=& -\frac{\nu k}{2}\log(2) - \frac{k (k-1)}{4} \log(\pi) - \sum_{i=1}^{k}{\log (\Gamma (\frac{\nu+1-i}{2}))}
           +\frac{\nu}{2} \log(\det(S)) - \frac{\nu+k+1}{2}\log (\det(W)) - \frac{1}{2} \mathsf{tr}(S W^{-1})
     \f}
     * 
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Inverse-Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_log(const Matrix<T_y,Dynamic,Dynamic>& W,
		    const T_dof& nu,
		    const Matrix<T_scale,Dynamic,Dynamic>& S) {
      return inv_wishart_log (W, nu, S, boost::math::policies::policy<>());
    }


    /**
     * The log of a density proportional to the Inverse-Wishart density for 
     * the given W, degrees of freedom, and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * nu must be greater than k-1
     *
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Inverse-Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_dof, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_propto_log(const Matrix<T_y,Dynamic,Dynamic>& W,
			   const T_dof& nu,
			   const Matrix<T_scale,Dynamic,Dynamic>& S,
			   const Policy& /* pol */) {
      return inv_wishart_log (W, nu, S, Policy());
    }
    
        /**
     * The log of a density proportional to the Inverse-Wishart density for 
     * the given W, degrees of freedom, and scale matrix. 
     * 
     * The scale matrix, S, must be k x k, symmetric, and semi-positive definite.
     * Dimension, k, is implicit.
     * nu must be greater than k-1
     *
     * @param W A scalar matrix
     * @param nu Degrees of freedom
     * @param S The scale matrix
     * @return The log of the Inverse-Wishart density at W given nu and S.
     * @throw std::domain_error if nu is not greater than k-1
     * @throw std::domain_error if S is not square, not symmetric, or not semi-positive definite.
     * @tparam T_y Type of scalar.
     * @tparam T_dof Type of degrees of freedom.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_dof, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_dof,T_scale>::type
    inv_wishart_propto_log(const Matrix<T_y,Dynamic,Dynamic>& W,
			   const T_dof& nu,
			   const Matrix<T_scale,Dynamic,Dynamic>& S) {
      return inv_wishart_propto_log (W, nu, S, boost::math::policies::policy<>());
    }



  }
}
#endif
