#ifndef __STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP__
#define __STAN__PROB__DISTRIBUTIONS__MULTI_STUDENT_T_HPP__

#include <cstdlib>

#include <stan/maths/matrix.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/error_handling.hpp>
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
      
      using boost::math::tools::promote_args;
      using std::isinf;


      // FIXME:  if nu = infinity, call multivariate normal

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

      // false means allow infinity
      if (!check_positive<false>(function, nu, 
                                 "Degrees of freedom, nu", &lp,
                                 Policy()))
        return lp;
      
      // FIXME: calls checks twice

      if (isinf(nu)) // already checked nu > 0
        return multi_normal_log(y,mu,Sigma,Policy());

      double d = y.size();

      lp += lgamma(0.5 * (nu + d));
      
      lp -= lgamma(0.5 * nu);
      
      lp -= (0.5 * d) * log(nu);
      
      lp -= (0.5 * d) * LOG_PI;

      using std::fabs;
      using stan::maths::determinant;
      using stan::maths::inverse;
      using stan::maths::multiply;
      using stan::maths::subtract;
      using stan::maths::transpose;

      lp -= 0.5 * log(fabs(determinant(Sigma)));

      Eigen::Matrix<T_scale,Eigen::Dynamic,1> y_minus_mu
        = subtract(y,mu);

      typename promote_args<T_y,T_loc,T_scale>::type temp
        = 

      lp -= 0.5 * (nu + d) 
        * log(1.0 + (multiply(multiply(transpose(y_minus_mu),
                                       inverse(Sigma)),
                              y_minus_mu)
                     / nu));

      return lp;
    }

  }
}
#endif
