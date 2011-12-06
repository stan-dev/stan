#ifndef __STAN__PROB__DISTRIBUTIONS_GAMMA_HPP__
#define __STAN__PROB__DISTRIBUTIONS_GAMMA_HPP__

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
     * The log of a gamma density for y with the specified
     * shape and inverse scale parameters.
     * Shape and inverse scale parameters must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Gamma}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\beta^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} \exp^{- \beta y} \right) \\
     &=& \alpha \log(\beta) - \log(\Gamma(\alpha)) + (\alpha - 1) \log(y) - \beta y\\
     & & \mathrm{where} \; y > 0
     \f}
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_shape, typename T_inv_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_inv_scale>::type
    gamma_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta, const Policy& /* pol */) {
      static const char* function = "stan::prob::gamma_log<%1%>(%1%)";

      double result;
      if (!stan::prob::check_positive(function, alpha, "Shape parameter", &result, Policy())) 
	return result;
      if (!stan::prob::check_positive(function, beta, "Inverse scale parameter", &result, Policy())) 
	return result;
      if (!stan::prob::check_nonnegative(function, y, "Random variate y", &result, Policy()))
	return result;

      return - lgamma(alpha)
	+ alpha * log(beta)
	+ (alpha - 1.0) * log(y)
	- beta * y;
    }

    /**
     * The log of a gamma density for y with the specified
     * shape and inverse scale parameters.
     * Shape and inverse scale parameters must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     \f{eqnarray*}{
     y &\sim& \mbox{\sf{Gamma}}(\alpha, \beta) \\
     \log (p (y \,|\, \alpha, \beta) ) &=& \log \left( \frac{\beta^\alpha}{\Gamma(\alpha)} y^{\alpha - 1} \exp^{- \beta y} \right) \\
     &=& \alpha \log(\beta) - \log(\Gamma(\alpha)) + (\alpha - 1) \log(y) - \beta y\\
     & & \mathrm{where} \; y > 0
     \f}
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_shape, typename T_inv_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_inv_scale>::type
    gamma_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta) {
      return gamma_log (y, alpha, beta, boost::math::policies::policy<>());
    }


    /**
     * The log of a distribution proportional to a gamma density for y with the specified
     * shape and inverse scale parameters.
     * Shape and inverse scale parameters must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_shape, typename T_inv_scale, class Policy>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_inv_scale>::type
    gamma_propto_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta, const Policy& /* pol */) {
      return gamma_log (y, alpha, beta, Policy());
    }
    

    /**
     * The log of a distribution proportional to a gamma density for y with the specified
     * shape and inverse scale parameters.
     * Shape and inverse scale parameters must be greater than 0.
     * y must be greater than or equal to 0.
     * 
     * @param y A scalar variable.
     * @param alpha Shape parameter.
     * @param beta Inverse scale parameter.
     * @throw std::domain_error if alpha is not greater than 0.
     * @throw std::domain_error if beta is not greater than 0.
     * @throw std::domain_error if y is not greater than or equal to 0.
     * @tparam T_y Type of scalar.
     * @tparam T_shape Type of shape.
     * @tparam T_inv_scale Type of inverse scale.
     */
    template <typename T_y, typename T_shape, typename T_inv_scale>
    inline typename boost::math::tools::promote_args<T_y,T_shape,T_inv_scale>::type
    gamma_propto_log(const T_y& y, const T_shape& alpha, const T_inv_scale& beta){
      return gamma_propto_log (y, alpha, beta, boost::math::policies::policy<> ());
    }
    
  }
}

#endif
