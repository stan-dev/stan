#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__WISHART_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__CONTINUOUS__WISHART_HPP

#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/matrix/check_ldlt_factor.hpp>
#include <stan/error_handling/scalar/check_greater.hpp>
#include <stan/math/functions/lmgamma.hpp>
#include <stan/math/matrix/crossprod.hpp>
#include <stan/math/matrix/columns_dot_product.hpp>
#include <stan/math/matrix/trace.hpp>
#include <stan/math/matrix/log_determinant_ldlt.hpp>
#include <stan/math/matrix/mdivide_left_ldlt.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/prob/distributions/univariate/continuous/chi_square.hpp>
#include <stan/prob/traits.hpp>

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
      static const std::string function("stan::prob::wishart_log");

      using boost::math::tools::promote_args;
      using Eigen::Dynamic;
      using Eigen::Lower;
      using Eigen::Matrix;
      using stan::error_handling::check_greater;
      using stan::error_handling::check_ldlt_factor;
      using stan::error_handling::check_size_match;
      using stan::math::index_type;
      using stan::math::LDLT_factor;
      using stan::math::log_determinant_ldlt;
      using stan::math::mdivide_left_ldlt;
      

      typename index_type<Matrix<T_scale,Dynamic,Dynamic> >::type k 
        = W.rows();
      typename promote_args<T_y,T_dof,T_scale>::type lp(0.0);
      check_greater(function, "Degrees of freedom parameter", nu, k-1);
      check_size_match(function, 
                       "Rows of random variable", W.rows(), 
                       "columns of random variable", W.cols());
      check_size_match(function, 
                       "Rows of scale parameter", S.rows(),
                       "columns of scale parameter", S.cols());
      check_size_match(function, 
                       "Rows of random variable", W.rows(), 
                       "columns of scale parameter", S.rows());
      // FIXME: domain checks

      LDLT_factor<T_y,Eigen::Dynamic,Eigen::Dynamic> ldlt_W(W);
      if (!check_ldlt_factor(function, "LDLT_Factor of random variable", ldlt_W))
        return lp;

      LDLT_factor<T_scale,Eigen::Dynamic,Eigen::Dynamic> ldlt_S(S);
      if (!check_ldlt_factor(function, "LDLT_Factor of scale parameter", ldlt_S))
        return lp;
      
      using stan::math::trace;
      using stan::math::lmgamma;
      if (include_summand<propto,T_dof>::value)
        lp += nu * k * NEG_LOG_TWO_OVER_TWO;

      if (include_summand<propto,T_dof>::value)
        lp -= lmgamma(k, 0.5 * nu);

      if (include_summand<propto,T_dof,T_scale>::value)
        lp -= 0.5 * nu * log_determinant_ldlt(ldlt_S);

      if (include_summand<propto,T_scale,T_y>::value) {
        Matrix<typename promote_args<T_y,T_scale>::type,Dynamic,Dynamic> 
          Sinv_W(mdivide_left_ldlt(ldlt_S, 
                                   static_cast<Matrix<T_y,Dynamic,Dynamic> >(W.template selfadjointView<Lower>())));
        lp -= 0.5 * trace(Sinv_W);
      }

      if (include_summand<propto,T_y,T_dof>::value && nu != (k + 1))
        lp += 0.5 * (nu - k - 1.0) * log_determinant_ldlt(ldlt_W);
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

      using Eigen::MatrixXd;
      using stan::math::index_type;
      using stan::error_handling::check_size_match;
      using stan::error_handling::check_positive;

      static const std::string function("stan::prob::wishart_rng");

      typename index_type<MatrixXd>::type k = S.rows();

      check_positive(function, "degrees of freedom", nu);
      check_size_match(function, 
                       "Rows of scale parameter", S.rows(), 
                       "columns of scale parameter", S.cols());

      MatrixXd B = MatrixXd::Zero(k, k);

      for (int j = 0; j < k; ++j) {
        for (int i = 0; i < j; ++i)
          B(i, j) = normal_rng(0, 1, rng);
        B(j,j) = std::sqrt(chi_square_rng(nu - j, rng));
      }
                
      return stan::math::crossprod(B * S.llt().matrixU());
    }


  }

}
#endif
