#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG_MIX_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG_MIX_HPP

#include <cmath>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/agrad/rev/functions/value_of.hpp>
#include <stan/math/functions/log_mix.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const fvar<T>& lambda1, const fvar<T>& lambda2) {
      using std::exp;
      using stan::math::log_mix;
      T theta_d = theta.val_;
      T lambda1_d = lambda1.val_;
      T lambda2_d = lambda2.val_;

      T d_theta(0);
      T d_lambda1(0);
      T d_lambda2(0);
      if (lambda1_d > lambda2_d) {
        T lam2_m_lam1 = lambda2_d - lambda1_d;
        T exp_lam2_m_lam1 = exp(lam2_m_lam1);
        T one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        T one_m_t = 1 - theta_d;
        T one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        T t_plus_one_m_t_prod_exp_lam2_m_lam1 
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
        T lam1_m_lam2 = lambda1_d - lambda2_d;
        T exp_lam1_m_lam2 = exp(lam1_m_lam2);
        T exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        T one_m_t = 1 - theta_d;
        T t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        T one_m_t_plus_t_prod_exp_lam1_m_lam2 
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
      return fvar<T>(log_mix(theta_d, lambda1_d, lambda2_d),
                     theta.d_ * d_theta + lambda1.d_ * d_lambda1
                     + lambda2.d_ * d_lambda2);
    }

    template <typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const fvar<T>& lambda1, const double lambda2) {
      using stan::math::log_mix;
      using std::exp;
      T theta_d = theta.val_;
      T lambda1_d = lambda1.val_;

      T d_theta(0);
      T d_lambda1(0);
      if (lambda1_d > lambda2) {
        T lam2_m_lam1 = lambda2 - lambda1_d;
        T exp_lam2_m_lam1 = exp(lam2_m_lam1);
        T one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        T one_m_t = 1 - theta_d;
        T one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        T t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_theta 
          = one_m_exp_lam2_m_lam1 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda1
          = theta_d
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        T lam1_m_lam2 = lambda1_d - lambda2;
        T exp_lam1_m_lam2 = exp(lam1_m_lam2);
        T exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        T one_m_t = 1 - theta_d;
        T t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        T one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_theta 
          = exp_lam1_m_lam2_m_1
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda1
          = t_prod_exp_lam1_m_lam2
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return fvar<T>(log_mix(theta_d, lambda1_d, lambda2),
                     theta.d_ * d_theta + lambda1.d_ * d_lambda1);
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const double lambda1,const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using std::exp;
      T theta_d = theta.val_;
      double lambda1_d = lambda1;
      T lambda2_d = lambda2.val_;

      T d_theta(0);
      T d_lambda2(0);
      if (lambda1_d > lambda2_d) {
        T lam2_m_lam1 = lambda2_d - lambda1_d;
        T exp_lam2_m_lam1 = exp(lam2_m_lam1);
        T one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        T one_m_t = 1 - theta_d;
        T one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        T t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_theta 
          = one_m_exp_lam2_m_lam1 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda2
          = one_m_t_prod_exp_lam2_m_lam1
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        T lam1_m_lam2 = lambda1_d - lambda2_d;
        T exp_lam1_m_lam2 = exp(lam1_m_lam2);
        T exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        T one_m_t = 1 - theta_d;
        T t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        T one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_theta 
          = exp_lam1_m_lam2_m_1
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda2
          = one_m_t
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return fvar<T>(log_mix(theta_d, lambda1_d, lambda2_d),
                     theta.d_ * d_theta + lambda2.d_ * d_lambda2);
    }

    template<typename T>
    inline
    fvar<T> 
    log_mix(const double theta, const fvar<T>& lambda1, const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta;
      T lambda1_d = lambda1.val_;
      T lambda2_d = lambda2.val_;

      T d_lambda1(0);
      T d_lambda2(0);
      if (lambda1_d > lambda2_d) {
        T lam2_m_lam1 = lambda2_d - lambda1_d;
        T exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_t = 1 - theta_d;
        T one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        T t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_lambda1
          = theta_d
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        d_lambda2
          = one_m_t_prod_exp_lam2_m_lam1
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        T lam1_m_lam2 = lambda1_d - lambda2_d;
        T exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double one_m_t = 1 - theta_d;
        T t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        T one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_lambda1
          = t_prod_exp_lam1_m_lam2
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
        d_lambda2
          = one_m_t
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return fvar<T>(log_mix(theta_d, lambda1_d, lambda2_d),
                     lambda1.d_ * d_lambda1 + lambda2.d_ * d_lambda2);
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const double lambda1, const double lambda2) {
      using stan::math::log_mix;
      using std::exp;
      T theta_d = theta.val_;
      double lambda1_d = lambda1;
      double lambda2_d = lambda2;

      T d_theta(0);
      if (lambda1_d > lambda2_d) {
        double lam2_m_lam1 = lambda2_d - lambda1_d;
        double exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_exp_lam2_m_lam1 = 1 - exp_lam2_m_lam1;
        T one_m_t = 1 - theta_d;
        T one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        T t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_theta 
          = one_m_exp_lam2_m_lam1 
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        double lam1_m_lam2 = lambda1_d - lambda2_d;
        double exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double exp_lam1_m_lam2_m_1 = exp_lam1_m_lam2 - 1;
        T one_m_t = 1 - theta_d;
        T t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        T one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_theta 
          = exp_lam1_m_lam2_m_1
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return fvar<T>(log_mix(theta_d, lambda1_d, lambda2_d),
                     theta.d_ * d_theta); 
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const double theta, const fvar<T>& lambda1, const double lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta;
      T lambda1_d = lambda1.val_;
      double lambda2_d = lambda2;

      T d_lambda1(0);
      if (lambda1_d > lambda2_d) {
        T lam2_m_lam1 = lambda2_d - lambda1_d;
        T exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_t = 1 - theta_d;
        T one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        T t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_lambda1
          = theta_d
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        T lam1_m_lam2 = lambda1_d - lambda2_d;
        T exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double one_m_t = 1 - theta_d;
        T t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        T one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_lambda1
          = t_prod_exp_lam1_m_lam2
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return fvar<T>(log_mix(theta_d, lambda1_d, lambda2_d),
                     lambda1.d_ * d_lambda1); 
    }

    template<typename T>
    inline
    fvar<T> 
    log_mix(const double theta, const double lambda1, const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using std::exp;
      double theta_d = theta;
      double lambda1_d = lambda1;
      T lambda2_d = lambda2.val_;

      T d_lambda2(0);
      if (lambda1_d > lambda2_d) {
        T lam2_m_lam1 = lambda2_d - lambda1_d;
        T exp_lam2_m_lam1 = exp(lam2_m_lam1);
        double one_m_t = 1 - theta_d;
        T one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        T t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        d_lambda2
          = one_m_t_prod_exp_lam2_m_lam1
          / t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } else {
        T lam1_m_lam2 = lambda1_d - lambda2_d;
        T exp_lam1_m_lam2 = exp(lam1_m_lam2);
        double one_m_t = 1 - theta_d;
        T t_prod_exp_lam1_m_lam2 = theta_d * exp_lam1_m_lam2;
        T one_m_t_plus_t_prod_exp_lam1_m_lam2 
          = one_m_t + t_prod_exp_lam1_m_lam2;
        d_lambda2
          = one_m_t
          / one_m_t_plus_t_prod_exp_lam1_m_lam2;
      }
      return fvar<T>(log_mix(theta_d, lambda1_d, lambda2_d),
                     lambda2.d_ * d_lambda2); 
    }
  }
}
#endif
