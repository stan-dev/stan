#ifndef __STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP__

#include <cstdlib>

#include <stan/prob/constants.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>

namespace stan {

  namespace prob {

    /**
     * Return the log of the multivariate Student t distribution
     * at the specified arguments.
     *
     * @tparam propto Carry out calculations up to a proportion
     */
    template <bool propto = false, 
              typename T_y, typename T_dof, typename T_loc, typename T_scale, 
              class Policy = boost::math::policies::policy<> > 
    inline 
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    multi_student_t_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                        const T_dof& nu,
                        const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                        const 
                        Eigen::Matrix<T_scale,
                                      Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                        const Policy& = Policy()) {
      static const char* function = "stan::prob::multi_student_t<%1%>(%1%)";

      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_cov_matrix;      
      using stan::math::check_positive;      
      using boost::math::tools::promote_args;

      typename promote_args<T_y,T_dof,T_loc,T_scale>::type lp(0.0);
      if (!check_size_match(function, y.size(), mu.size(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, y.size(), Sigma.rows(), &lp, Policy()))
        return lp;
      if (!check_size_match(function, y.size(), Sigma.cols(), &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter, mu", &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "y", &lp, Policy())) 
        return lp;
      if (!check_cov_matrix(function, Sigma, &lp, Policy()))
        return lp;

      // allows infinities
      if (!check_not_nan(function, nu, 
                         "Degrees of freedom, nu", &lp,
                         Policy()))
        return lp;
      if (!check_positive(function, nu, 
                          "Degrees of freedom, nu", &lp,
                          Policy()))
        return lp;
      
      // FIXME: calls expensive (!) checks twice, here and in multi normal
      using std::isinf;

      if (isinf(nu)) // already checked nu > 0
        return multi_normal_log(y,mu,Sigma,Policy());

      double d = y.size();

      if (include_summand<propto,T_dof>::value) {
        lp += lgamma(0.5 * (nu + d));
        lp -= lgamma(0.5 * nu);
        lp -= (0.5 * d) * log(nu);
      }

      if (include_summand<propto>::value) 
        lp -= (0.5 * d) * LOG_PI;

      using std::fabs;
      using stan::math::determinant;
      using stan::math::inverse;
      using stan::math::multiply;
      using stan::math::subtract;
      using stan::math::transpose;

      if (include_summand<propto,T_scale>::value) 
        lp -= 0.5 * log(fabs(determinant(Sigma)));

      if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value) {
        Eigen::Matrix<typename promote_args<T_y,T_loc>::type,
                      Eigen::Dynamic,
                      1> y_minus_mu = subtract(y,mu);
        
        lp -= 0.5 
          * (nu + d) 
          * log(1.0 + (multiply(multiply(transpose(y_minus_mu),
                                         inverse(Sigma)),
                                y_minus_mu)
                       / nu));
      }
      return lp;
    }

  }
}
#endif
