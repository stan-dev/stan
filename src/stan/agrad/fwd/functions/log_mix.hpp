#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG_MIX_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG_MIX_HPP

#include <cmath>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/agrad/rev/functions/value_of.hpp>
#include <stan/math/functions/log_mix.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan {

  namespace agrad {
    using boost::math::tools::promote_args;
    template <typename T1, typename T2>
    struct mult_wrapper {
      template <typename T3, typename T4, int N>
      static void mult(const T3& num1, const T4& num2, 
                       typename promote_args<T3, T4>::type (&vec_inp)[N],
                       unsigned int& offset){
      } 
    };
    template <typename T1>
    struct mult_wrapper<T1, T1> {
      template <typename T3, typename T4, int N>
      static void mult(const T3& num1, const T4& num2, 
                       typename promote_args<T3, T4>::type (&vec_inp)[N],
                       unsigned int& offset){
        vec_inp[offset] = num1 * num2;
        ++offset;
      }
    };

    template <typename t_T, typename l1_T, typename l2_T, int N>
    inline void
    log_mix_calc(const t_T& theta_d, 
                 const l1_T& lambda1_d, 
                 const l2_T& lambda2_d,
                 typename promote_args<t_T,l1_T,l2_T>::type (&ret_vec)[N]){
        typedef typename promote_args<t_T,l1_T,l2_T>::type dom_arg_type;
        using std::exp;

        typename promote_args<l1_T,l2_T>::type lam2_m_lam1 = lambda2_d - lambda1_d;
        typename promote_args<l1_T,l2_T>::type exp_lam2_m_lam1 = exp(lam2_m_lam1);
        typename promote_args<l1_T,l2_T>::type one_m_exp_lam2_m_lam1 = 1.0 - exp_lam2_m_lam1;
        typename promote_args<double,t_T>::type one_m_t = 1.0 - theta_d;
        dom_arg_type one_m_t_prod_exp_lam2_m_lam1 = one_m_t * exp_lam2_m_lam1;
        dom_arg_type t_plus_one_m_t_prod_exp_lam2_m_lam1 
          = theta_d + one_m_t_prod_exp_lam2_m_lam1;
        dom_arg_type one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1
          = 1.0 / t_plus_one_m_t_prod_exp_lam2_m_lam1;
        
        unsigned int offset = 0;
        mult_wrapper<t_T, dom_arg_type>::mult(one_m_exp_lam2_m_lam1,
                                         one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1,
                                         ret_vec, offset);
        mult_wrapper<l1_T, dom_arg_type>::mult(theta_d,
                                          one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1,
                                          ret_vec, offset);
        mult_wrapper<l2_T, dom_arg_type>::mult(one_m_t_prod_exp_lam2_m_lam1,
                                          one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1,
                                          ret_vec, offset);
    }


    template <typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const fvar<T>& lambda1, const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1.val_ > lambda2.val_) {
        fvar<T> vec_T[3];
        log_mix_calc(theta, lambda1, lambda2, vec_T);
        return fvar<T>(log_mix(theta.val_, lambda1.val_, lambda2.val_),
                       theta.d_ * value_of(vec_T[0]) + lambda1.d_ * value_of(vec_T[1]) 
                       + lambda2.d_ * value_of(vec_T[2]));
      } else {
        fvar<T> vec_T[3];
        log_mix_calc(1.0 - theta, lambda2, lambda1, vec_T);
        return fvar<T>(log_mix(theta.val_, lambda1.val_, lambda2.val_),
                       theta.d_ * -1.0 * value_of(vec_T[0]) 
                       + lambda1.d_ * value_of(vec_T[2]) 
                       + lambda2.d_ * value_of(vec_T[1]));
      }
    }

    template <typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const fvar<T>& lambda1, const double lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1.val_ > lambda2) {
        fvar<T> vec_T[2];
        log_mix_calc(theta, lambda1, lambda2, vec_T);
        return fvar<T>(log_mix(theta.val_, lambda1.val_, lambda2),
                       theta.d_ * value_of(vec_T[0]) 
                       + lambda1.d_ * value_of(vec_T[1]));
      } else {
        fvar<T> vec_T[2];
        log_mix_calc(1.0 - theta, lambda2, lambda1, vec_T);
        return fvar<T>(log_mix(theta.val_, lambda1.val_, lambda2),
                       -1.0 * theta.d_ * value_of(vec_T[0]) 
                       + lambda1.d_ * value_of(vec_T[1]));
      }
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const double lambda1,const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1 > lambda2.val_) {
        fvar<T> vec_T[2];
        log_mix_calc(theta, lambda1, lambda2,vec_T);
        return fvar<T>(log_mix(theta.val_, lambda1, lambda2.val_),
                       theta.d_ * value_of(vec_T[0]) 
                       + lambda2.d_ * value_of(vec_T[1]));
      } else {
        fvar<T> vec_T[2];
        log_mix_calc(1.0 - theta, lambda2, lambda1,vec_T);
        return fvar<T>(log_mix(theta.val_, lambda1, lambda2.val_),
                       -1.0 * theta.d_ * value_of(vec_T[0]) 
                       + lambda2.d_ * value_of(vec_T[1]));
      }
    }

    template<typename T>
    inline
    fvar<T> 
    log_mix(const double theta, const fvar<T>& lambda1, const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1.val_ > lambda2.val_) {
        fvar<T> vec_T[2];
        log_mix_calc(theta, lambda1, lambda2, vec_T);
        return fvar<T>(log_mix(theta, lambda1.val_, lambda2.val_),
                       lambda1.d_ * value_of(vec_T[0]) 
                       + lambda2.d_ * value_of(vec_T[1]));
      } else {
        fvar<T> vec_T[2];
        log_mix_calc(1.0 - theta, lambda2, lambda1,vec_T);
        return fvar<T>(log_mix(theta, lambda1.val_, lambda2.val_),
                       lambda1.d_ * value_of(vec_T[1]) 
                       + lambda2.d_ * value_of(vec_T[0]));
      }
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const double lambda1, const double lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1 > lambda2) {
        fvar<T> vec_T[1];
        log_mix_calc(theta, lambda1, lambda2, vec_T);
        return fvar<T>(log_mix(theta.val_, lambda1, lambda2),
                       theta.d_ * value_of(vec_T[0]));
      } else {
        fvar<T> vec_T[1];
        log_mix_calc(1.0 - theta, lambda2, lambda1,vec_T);
        return fvar<T>(log_mix(theta.val_, lambda1, lambda2),
                       -1.0 * theta.d_ * value_of(vec_T[0]));
      }
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const double theta, const fvar<T>& lambda1, const double lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;
      
      if (lambda1.val_ > lambda2) {
        fvar<T> vec_T[1];
        log_mix_calc(theta, lambda1, lambda2, vec_T);
        return fvar<T>(log_mix(theta, lambda1.val_, lambda2),
                       lambda1.d_ * value_of(vec_T[0]));
      } else {
        fvar<T> vec_T[1];
        log_mix_calc(1.0 - theta, lambda2, lambda1, vec_T);
        return fvar<T>(log_mix(theta, lambda1.val_, lambda2),
                       lambda1.d_ * value_of(vec_T[0]));
      }
    }

    template<typename T>
    inline
    fvar<T> 
    log_mix(const double theta, const double lambda1, const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1 > lambda2.val_) {
        fvar<T> vec_T[1];
        log_mix_calc(theta, lambda1, lambda2, vec_T);
        return fvar<T>(log_mix(theta, lambda1, lambda2.val_),
                       lambda2.d_ * value_of(vec_T[0]));
      } else {
        fvar<T> vec_T[1];
        log_mix_calc(1.0 - theta, lambda2, lambda1,vec_T);
        return fvar<T>(log_mix(theta, lambda1, lambda2.val_),
                       lambda2.d_ * value_of(vec_T[0]));
        }
    }
  }
}
#endif
