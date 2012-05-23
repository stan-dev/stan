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
    template <bool propto,
              typename T_y, typename T_dof, typename T_loc, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    multi_student_t_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                        const T_dof& nu,
                        const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                        const 
                        Eigen::Matrix<T_scale,
                                      Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                        const Policy&) {
      static const char* function = "stan::prob::multi_student_t<%1%>(%1%)";

      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
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
      if (!check_symmetric(function, Sigma, "Sigma", &lp, Policy()))
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
      
      using std::isinf;

      if (isinf(nu)) // already checked nu > 0
        return multi_normal_log(y,mu,Sigma,Policy());

      Eigen::LLT< Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic> > LLT = Sigma.llt();
      if (LLT.info() != Eigen::Success) {
        lp = stan::math::policies::raise_domain_error<T_scale>(function,
                                              "Sigma is not positive definite (%1%)",
                                              0,Policy());
        return lp;
      }
      Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic> L = LLT.matrixL();
      
      double d = y.size();

      if (include_summand<propto,T_dof>::value) {
        lp += lgamma(0.5 * (nu + d));
        lp -= lgamma(0.5 * nu);
        lp -= (0.5 * d) * log(nu);
      }

      if (include_summand<propto>::value) 
        lp -= (0.5 * d) * LOG_PI;

      using stan::math::multiply;
      using stan::math::dot_self;
      using stan::math::subtract;
      using Eigen::Array;
      using stan::math::mdivide_left_tri;


      if (include_summand<propto,T_scale>::value)
        lp -= L.diagonal().array().log().sum();

      if (include_summand<propto,T_y,T_dof,T_loc,T_scale>::value) {
//      Eigen::Matrix<T_scale,Eigen::Dynamic,Eigen::Dynamic> I(d,d);
//      I.setIdentity();
        
        Eigen::Matrix<typename promote_args<T_y,T_loc>::type,
                      Eigen::Dynamic,
                      1> y_minus_mu = subtract(y,mu);
        Eigen::Matrix<typename promote_args<T_scale,T_y,T_loc>::type,
                      Eigen::Dynamic,
                      1> half = L = mdivide_left_tri<Eigen::Lower>(L, y_minus_mu);
        lp -= 0.5 
          * (nu + d)
          * log(1.0 + dot_self(half) / nu);
      }
      return lp;
    }

    template <bool propto,
              typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline 
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    multi_student_t_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                        const T_dof& nu,
                        const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                        const 
                        Eigen::Matrix<T_scale,
                                      Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_student_t_log<propto>(y,nu,mu,Sigma,
                                         stan::math::default_policy());
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale, 
              class Policy>
    inline 
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    multi_student_t_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                        const T_dof& nu,
                        const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                        const 
                        Eigen::Matrix<T_scale,
                                      Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                        const Policy&) {
      return multi_student_t_log<false>(y,nu,mu,Sigma,Policy());
    }



    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline 
    typename boost::math::tools::promote_args<T_y,T_dof,T_loc,T_scale>::type
    multi_student_t_log(const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                        const T_dof& nu,
                        const Eigen::Matrix<T_loc,Eigen::Dynamic,1>& mu,
                        const 
                        Eigen::Matrix<T_scale,
                                      Eigen::Dynamic,Eigen::Dynamic>& Sigma) {
      return multi_student_t_log<false>(y,nu,mu,Sigma,
                                         stan::math::default_policy());
    }



  }
}
#endif
