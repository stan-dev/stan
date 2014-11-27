#ifndef STAN__AGRAD__REV__FUNCTIONS__LOG_MIX_HPP
#define STAN__AGRAD__REV__FUNCTIONS__LOG_MIX_HPP

#include <cmath>
#include <stan/agrad/rev.hpp>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>
#include <stan/math/functions/log_sum_exp.hpp>
#include <stan/math/functions/log1m.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/agrad/rev/functions/value_of.hpp>
#include <stan/meta/traits.hpp>
#include <stan/agrad/partials_vari.hpp>


namespace stan {

  namespace agrad {

    inline 
    void log_mix_deriv_inter_helper(const double& theta_val,
                                    const double& lambda1_val,
                                    const double& lambda2_val,
                                    double& one_m_exp_lam2_m_lam1,
                                    double& one_m_t_prod_exp_lam2_m_lam1,
                                    double& one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1){
        using ::exp;
        double lam2_m_lam1 = lambda2_val - lambda1_val;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        double one_m_t = 1 - theta_val;
        one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = 1 / (theta_val + one_m_t_prod_exp_lam2_m_lam1);
    }

    template <typename T_theta,
              typename T_lambda1,
              typename T_lambda2>
    inline 
    typename return_type<T_theta, T_lambda1, T_lambda2>::type
    log_mix(const T_theta& theta,
            const T_lambda1& lambda1,
            const T_lambda2& lambda2){
      using ::log;
      using stan::math::log_sum_exp;
      using stan::math::log1m;
      using stan::is_constant_struct;

      stan::error_handling::check_bounded("log_mix","theta",theta,0,1);
      stan::error_handling::check_not_nan("log_mix","lambda1",lambda1);
      stan::error_handling::check_not_nan("log_mix","lambda2",lambda2);

      OperandsAndPartials<T_theta, T_lambda1, T_lambda2> 
        operands_and_partials(theta, lambda1, lambda2);

      double theta_double = value_of(theta);
      const double lambda1_double = value_of(lambda1);
      const double lambda2_double = value_of(lambda2);

      double log_mix_function_value 
        = log_sum_exp(log(theta_double) + lambda1_double,
                         log1m(theta_double) + lambda2_double);

      double one_m_exp_lam2_m_lam1(0.0); 
      double one_m_t_prod_exp_lam2_m_lam1(0.0);
      double one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1(0.0);

      if (lambda1 > lambda2)
        log_mix_deriv_inter_helper(theta_double, 
                                   lambda1_double, 
                                   lambda2_double,
                                   one_m_exp_lam2_m_lam1,
                                   one_m_t_prod_exp_lam2_m_lam1,
                                   one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1);
      else {
        log_mix_deriv_inter_helper(1.0 - theta_double, 
                                   lambda2_double, 
                                   lambda1_double,
                                   one_m_exp_lam2_m_lam1,
                                   one_m_t_prod_exp_lam2_m_lam1,
                                   one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1);
        one_m_exp_lam2_m_lam1 *= -1.0;
        theta_double = one_m_t_prod_exp_lam2_m_lam1;
        one_m_t_prod_exp_lam2_m_lam1 = 1.0 - value_of(theta);
      }

      if (!is_constant_struct<T_theta>::value)
        operands_and_partials.d_x1[0] 
          = one_m_exp_lam2_m_lam1
          * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;
      if (!is_constant_struct<T_lambda1>::value)
        operands_and_partials.d_x2[0] 
          = theta_double 
          * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;
      if (!is_constant_struct<T_lambda2>::value)
        operands_and_partials.d_x3[0] 
          = one_m_t_prod_exp_lam2_m_lam1 
          * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;

      return operands_and_partials.to_var(log_mix_function_value);
    }

  } // namespace agrad

} // namespace stan

#endif
