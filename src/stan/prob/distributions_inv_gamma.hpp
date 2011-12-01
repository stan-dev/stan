#ifndef __STAN__PROB__DISTRIBUTIONS_INV_GAMMA_HPP__
#define __STAN__PROB__DISTRIBUTIONS_INV_GAMMA_HPP__

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

    /**
     * The log of an inverse gamma density for y with the specified
     * shape and scale parameters.
     * Shape and scale parameters must be greater than 0.
     * y must be greater than 0.
     * 
     \f{eqnarray*}{
       y &\sim& \mbox{\sf{Inv-gamma}}(\alpha, \beta) \\
       \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\beta^\alpha}{\Gamma(\alpha)} y^{-(\alpha + 1)} \exp^{- \beta / y} \right) \\
       &=& \alpha \log(\beta) - \log(\Gamma(\alpha)) - (\alpha + 1) \log(y) - \frac{\beta}{y} \\
       & & \mathrm{where } y > 0
     \f}
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_shape, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta, const Policy& /* pol */) {
      static const char* function = "stan::prob::inv_gamma_log<%1%>(%1%)";

      typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type result;
      if (false == stan::prob::check_positive(function, alpha, "Shape parameter", &result, Policy())) 
	return result;
      if (false == stan::prob::check_positive(function, beta, "Scale parameter", &result, Policy())) 
	return result;
      if (false == stan::prob::check_positive(function, y, "Random variate y", &result, Policy()))
	return result;
      return - lgamma(alpha)
	+ alpha * log(beta)
	- (alpha + 1) * log(y)
	- beta / y;
    }
    /**
     * The log of an inverse gamma density for y with the specified
     * shape and scale parameters.
     * Shape and scale parameters must be greater than 0.
     * y must be greater than 0.
     * 
     \f{eqnarray*}{
       y &\sim& \mbox{\sf{Inv-gamma}}(\alpha, \beta) \\
       \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\beta^\alpha}{\Gamma(\alpha)} y^{-(\alpha + 1)} \exp^{- \beta / y} \right) \\
       &=& \alpha \log(\beta) - \log(\Gamma(\alpha)) - (\alpha + 1) \log(y) - \frac{\beta}{y} \\
       & & \mathrm{where } y > 0
     \f}
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_shape, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_log(const T_y& y, const T_shape& alpha, const T_scale& beta) {
      return inv_gamma_log (y, alpha, beta, boost::math::policies::policy<>());
    }
    /**
     * The log of a distribution proportional to an inverse gamma density for y with the specified
     * shape and scale parameters.
     * Shape and scale parameters must be greater than 0.
     * y must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_shape, typename T_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_propto_log(const T_y& y, const T_shape& alpha, const T_scale& beta, const Policy& /* pol */) {
      return inv_gamma_log (y, alpha, beta, Policy());
    }
    /**
     * The log of a distribution proportional to an inverse gamma density for y with the specified
     * shape and scale parameters.
     * Shape and scale parameters must be greater than 0.
     * y must be greater than 0.
     * 
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_scale Type of scale.
     */
    template <typename T_y, typename T_shape, typename T_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_scale>::type
    inv_gamma_propto_log(const T_y& y, const T_shape& alpha, const T_scale& beta) {
      return inv_gamma_log (y, alpha, beta, boost::math::policies::policy<>());
    }
    
  }
}

#endif
