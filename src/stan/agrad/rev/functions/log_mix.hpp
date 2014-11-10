#ifndef STAN__AGRAD__REV__FUNCTIONS__LOG_MIX_HPP
#define STAN__AGRAD__REV__FUNCTIONS__LOG_MIX_HPP

#include <cmath>
#include <stan/agrad/rev.hpp>
#include <stan/math/functions/log_mix.hpp>

namespace stan {

  namespace agrad {

    var log_mix(var theta, var lambda1, var lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta.val();
      double lambda1_d = lambda1.val();
      double lambda2_d = lambda2.val();
      double result = stan::math::log_mix(theta_d, lambda1_d, lambda2_d);

      double d_theta(0);
      double d_lambda1(0);
      double d_lambda2(0);
      if (lambda1_d > lambda2_d) {
        double lam2_m_lam1 = lambda2_d - lambda1_d;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        double one_m_t = 1 - theta_d;
        double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        double t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_theta 
          = one_m_exp_lam2_m_lam1 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda1
          = theta_d
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda2
          = one_m_t_prod_exp_lam2_m_lam1
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        double lam1_m_lam2 = lambda1_d - lambda2_d;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        double one_m_t = 1 - theta_d;
        double t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        double one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_theta 
          = exp_lam1_m_lam2_m_1
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda1
          = t_prod_exp_lam1_m_lam2
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda2
          = one_m_t
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return var(new precomp_vvv_vari(result, 
                                      theta.vi_, lambda1.vi_, lambda2.vi_,
                                      d_theta, d_lambda1, d_lambda2));
    }

    var log_mix(var theta, var lambda1, double lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta.val();
      double lambda1_d = lambda1.val();
      double result = stan::math::log_mix(theta_d, lambda1_d, lambda2);

      double d_theta(0);
      double d_lambda1(0);
      if (lambda1_d > lambda2) {
        double lam2_m_lam1 = lambda2 - lambda1_d;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        double one_m_t = 1 - theta_d;
        double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        double t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_theta 
          = one_m_exp_lam2_m_lam1 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda1
          = theta_d
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        double lam1_m_lam2 = lambda1_d - lambda2;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        double one_m_t = 1 - theta_d;
        double t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        double one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_theta 
          = exp_lam1_m_lam2_m_1
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda1
          = t_prod_exp_lam1_m_lam2
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return var(new precomp_vv_vari(result, 
                                      theta.vi_, lambda1.vi_, 
                                      d_theta, d_lambda1));
    }

    var log_mix(var theta, double lambda1, var lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta.val();
      double lambda1_d = lambda1;
      double lambda2_d = lambda2.val();
      double result = stan::math::log_mix(theta_d, lambda1_d, lambda2_d);

      double d_theta(0);
      double d_lambda2(0);
      if (lambda1_d > lambda2_d) {
        double lam2_m_lam1 = lambda2_d - lambda1_d;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        double one_m_t = 1 - theta_d;
        double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        double t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_theta 
          = one_m_exp_lam2_m_lam1 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda2
          = one_m_t_prod_exp_lam2_m_lam1
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        double lam1_m_lam2 = lambda1_d - lambda2_d;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        double one_m_t = 1 - theta_d;
        double t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        double one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_theta 
          = exp_lam1_m_lam2_m_1
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda2
          = one_m_t
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return var(new precomp_vv_vari(result, 
                                      theta.vi_, lambda2.vi_,
                                      d_theta, d_lambda2));
    }

    var log_mix(double theta, var lambda1, var lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta;
      double lambda1_d = lambda1.val();
      double lambda2_d = lambda2.val();
      double result = stan::math::log_mix(theta_d, lambda1_d, lambda2_d);

      double d_lambda1(0);
      double d_lambda2(0);
      if (lambda1_d > lambda2_d) {
        double lam2_m_lam1 = lambda2_d - lambda1_d;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_t = 1 - theta_d;
        double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        double t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_lambda1
          = theta_d
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda2
          = one_m_t_prod_exp_lam2_m_lam1
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        double lam1_m_lam2 = lambda1_d - lambda2_d;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double one_m_t = 1 - theta_d;
        double t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        double one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_lambda1
          = t_prod_exp_lam1_m_lam2
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda2
          = one_m_t
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return var(new precomp_vv_vari(result, 
                                      lambda1.vi_, lambda2.vi_,
                                      d_lambda1, d_lambda2));
    }

    var log_mix(var theta, double lambda1, double lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta.val();
      double lambda1_d = lambda1;
      double lambda2_d = lambda2;
      double result = stan::math::log_mix(theta_d, lambda1_d, lambda2_d);

      double d_theta(0);
      if (lambda1_d > lambda2_d) {
        double lam2_m_lam1 = lambda2_d - lambda1_d;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        double one_m_t = 1 - theta_d;
        double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        double t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_theta 
          = one_m_exp_lam2_m_lam1 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        double lam1_m_lam2 = lambda1_d - lambda2_d;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        double one_m_t = 1 - theta_d;
        double t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        double one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_theta 
          = exp_lam1_m_lam2_m_1
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return var(new precomp_v_vari(result, theta.vi_, d_theta));
    }

    var log_mix(double theta, var lambda1, double lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta;
      double lambda1_d = lambda1.val();
      double lambda2_d = lambda2;
      double result = stan::math::log_mix(theta_d, lambda1_d, lambda2_d);

      double d_lambda1(0);
      if (lambda1_d > lambda2_d) {
        double lam2_m_lam1 = lambda2_d - lambda1_d;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_t = 1 - theta_d;
        double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        double t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_lambda1
          = theta_d
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        double lam1_m_lam2 = lambda1_d - lambda2_d;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double one_m_t = 1 - theta_d;
        double t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        double one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_lambda1
          = t_prod_exp_lam1_m_lam2
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return var(new precomp_v_vari(result, lambda1.vi_, d_lambda1));
    }

    var log_mix(double theta, double lambda1, var lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta;
      double lambda1_d = lambda1;
      double lambda2_d = lambda2.val();
      double result = stan::math::log_mix(theta_d, lambda1_d, lambda2_d);

      double d_lambda2(0);
      if (lambda1_d > lambda2_d) {
        double lam2_m_lam1 = lambda2_d - lambda1_d;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_t = 1 - theta_d;
        double one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        double t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_lambda2
          = one_m_t_prod_exp_lam2_m_lam1
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        double lam1_m_lam2 = lambda1_d - lambda2_d;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double one_m_t = 1 - theta_d;
        double t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        double one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_lambda2
          = one_m_t
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return var(new precomp_v_vari(result, lambda2.vi_, d_lambda2));
    }


  } // namespace agrad

} // namespace stan

#endif
