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
		using boost::is_same;

    // log_mix is a function of 3 variables
    // The number of partial derivatives that need to be calculated is a function of
    // the types of the arguments. These are templated out in the
    // log_mix_helper_function that only returns the required partials 
    // It does this by sequentially checking the types of the arguments
    // theta_d, lambda1_d and lambda2_d against
    // the dominant argument type, dom_arg_type, which will always default to
    // fvar<T> given that this function shouldn't be called with three double
    // arguments. There exists a stan/src/stan/math/functions/log_mix.hpp file
    // to handle the triple double argument call, which necessarily returns
    // only thet value of the log_sum_exp(log(theta) + lambda_1, log(1 - theta) +
    // lambda_2). The partials are returned through the variable length n array
    // partials_array. 
    
    // In order to be computationally stable, we multiply each of the partial
    // derivatives by exp(max(lambda_1,lambda_2)) / exp(max(lambda_1,lambda_2))
    // The "max" calculation is handled by an if statement internal to each
    // implementation of log_mix.
    template <typename t_T, typename l1_T, typename l2_T, int N>
    inline void
    log_mix_partial_helper(const t_T& theta_d, 
                           const l1_T& lambda1_d, 
                           const l2_T& lambda2_d,
                           typename promote_args<t_T,l1_T,l2_T>::type (&partials_array)[N]){
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
      if (is_same<t_T, dom_arg_type>::value){
        partials_array[offset] 
          = one_m_exp_lam2_m_lam1 
          * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;
         ++offset;
      }
      if (is_same<l1_T, dom_arg_type>::value){
        partials_array[offset] 
          = theta_d * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;
         ++offset;
      }
      if (is_same<l2_T, dom_arg_type>::value){
        partials_array[offset] 
          = one_m_t_prod_exp_lam2_m_lam1 
          * one_d_t_plus_one_m_t_prod_exp_lam2_m_lam1;
      } 
    }

    /**
     * Return the log mixture density with specified mixing proportion
     * and log densities and its derivative at each.
     *
     * \f[
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2) 
     * = \log \left( \theta \lambda_1 + (1 - \theta) \lambda_2 \right).
     * \f]
     * 
     * \f[
     * \frac{\partial}{\partial \theta} 
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = \dfrac{\exp(\lambda_1) - \exp(\lambda_2)}
     * {\left( \theta \lambda_1 + (1 - \theta) \lambda_2 \right)}
     * \f]
     *
     * \f[
     * \frac{\partial}{\partial \lambda_1} 
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = \dfrac{\theta \exp(\lambda_1)}
     * {\left( \theta \lambda_1 + (1 - \theta) \lambda_2 \right)} 
     * \f]
     * 
     * \f[
     * \frac{\partial}{\partial \lambda_2} 
     * \mbox{log\_mix}(\theta, \lambda_1, \lambda_2)
     * = \dfrac{\theta \exp(\lambda_2)}
     * {\left( \theta \lambda_1 + (1 - \theta) \lambda_2 \right)} 
     * \f]
     * 
     * @param theta[in] mixing proportion in [0,1].
     * @param lambda1 first log density.
     * @param lambda2 second log density.
     * @return log mixture of densities in specified proportion
     */

    template <typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const fvar<T>& lambda1, const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1.val_ > lambda2.val_) {
        fvar<T> partial_deriv_array[3];
        log_mix_partial_helper(theta, lambda1, lambda2, partial_deriv_array);
        return fvar<T>(log_mix(theta.val_, lambda1.val_, lambda2.val_),
                       theta.d_ * value_of(partial_deriv_array[0]) 
                       + lambda1.d_ * value_of(partial_deriv_array[1]) 
                       + lambda2.d_ * value_of(partial_deriv_array[2]));
      } else {
        fvar<T> partial_deriv_array[3];
        log_mix_partial_helper(1.0 - theta, lambda2, lambda1, partial_deriv_array);
        return fvar<T>(log_mix(theta.val_, lambda1.val_, lambda2.val_),
                       theta.d_ * -1.0 * value_of(partial_deriv_array[0]) 
                       + lambda1.d_ * value_of(partial_deriv_array[2]) 
                       + lambda2.d_ * value_of(partial_deriv_array[1]));
      }
    }

    template <typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const fvar<T>& lambda1, const double lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1.val_ > lambda2) {
        fvar<T> partial_deriv_array[2];
        log_mix_partial_helper(theta, lambda1, lambda2, partial_deriv_array);
        return fvar<T>(log_mix(theta.val_, lambda1.val_, lambda2),
                       theta.d_ * value_of(partial_deriv_array[0]) 
                       + lambda1.d_ * value_of(partial_deriv_array[1]));
      } else {
        fvar<T> partial_deriv_array[2];
        log_mix_partial_helper(1.0 - theta, lambda2, lambda1, partial_deriv_array);
        return fvar<T>(log_mix(theta.val_, lambda1.val_, lambda2),
                       -1.0 * theta.d_ * value_of(partial_deriv_array[0]) 
                       + lambda1.d_ * value_of(partial_deriv_array[1]));
      }
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const double lambda1,const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1 > lambda2.val_) {
        fvar<T> partial_deriv_array[2];
        log_mix_partial_helper(theta, lambda1, lambda2, partial_deriv_array);
        return fvar<T>(log_mix(theta.val_, lambda1, lambda2.val_),
                       theta.d_ * value_of(partial_deriv_array[0]) 
                       + lambda2.d_ * value_of(partial_deriv_array[1]));
      } else {
        fvar<T> partial_deriv_array[2];
        log_mix_partial_helper(1.0 - theta, lambda2, lambda1, partial_deriv_array);
        return fvar<T>(log_mix(theta.val_, lambda1, lambda2.val_),
                       -1.0 * theta.d_ * value_of(partial_deriv_array[0]) 
                       + lambda2.d_ * value_of(partial_deriv_array[1]));
      }
    }

    template<typename T>
    inline
    fvar<T> 
    log_mix(const double theta, const fvar<T>& lambda1, const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1.val_ > lambda2.val_) {
        fvar<T> partial_deriv_array[2];
        log_mix_partial_helper(theta, lambda1, lambda2, partial_deriv_array);
        return fvar<T>(log_mix(theta, lambda1.val_, lambda2.val_),
                       lambda1.d_ * value_of(partial_deriv_array[0]) 
                       + lambda2.d_ * value_of(partial_deriv_array[1]));
      } else {
        fvar<T> partial_deriv_array[2];
        log_mix_partial_helper(1.0 - theta, lambda2, lambda1,partial_deriv_array);
        return fvar<T>(log_mix(theta, lambda1.val_, lambda2.val_),
                       lambda1.d_ * value_of(partial_deriv_array[1]) 
                       + lambda2.d_ * value_of(partial_deriv_array[0]));
      }
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const fvar<T>& theta, const double lambda1, const double lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1 > lambda2) {
        fvar<T> partial_deriv_array[1];
        log_mix_partial_helper(theta, lambda1, lambda2, partial_deriv_array);
        return fvar<T>(log_mix(theta.val_, lambda1, lambda2),
                       theta.d_ * value_of(partial_deriv_array[0]));
      } else {
        fvar<T> partial_deriv_array[1];
        log_mix_partial_helper(1.0 - theta, lambda2, lambda1,partial_deriv_array);
        return fvar<T>(log_mix(theta.val_, lambda1, lambda2),
                       -1.0 * theta.d_ * value_of(partial_deriv_array[0]));
      }
    }

    template<typename T>
    inline
    fvar<T>
    log_mix(const double theta, const fvar<T>& lambda1, const double lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;
      
      if (lambda1.val_ > lambda2) {
        fvar<T> partial_deriv_array[1];
        log_mix_partial_helper(theta, lambda1, lambda2, partial_deriv_array);
        return fvar<T>(log_mix(theta, lambda1.val_, lambda2),
                       lambda1.d_ * value_of(partial_deriv_array[0]));
      } else {
        fvar<T> partial_deriv_array[1];
        log_mix_partial_helper(1.0 - theta, lambda2, lambda1, partial_deriv_array);
        return fvar<T>(log_mix(theta, lambda1.val_, lambda2),
                       lambda1.d_ * value_of(partial_deriv_array[0]));
      }
    }

    template<typename T>
    inline
    fvar<T> 
    log_mix(const double theta, const double lambda1, const fvar<T>& lambda2) {
      using stan::math::log_mix;
      using stan::math::value_of;

      if (lambda1 > lambda2.val_) {
        fvar<T> partial_deriv_array[1];
        log_mix_partial_helper(theta, lambda1, lambda2, partial_deriv_array);
        return fvar<T>(log_mix(theta, lambda1, lambda2.val_),
                       lambda2.d_ * value_of(partial_deriv_array[0]));
      } else {
        fvar<T> partial_deriv_array[1];
        log_mix_partial_helper(1.0 - theta, lambda2, lambda1, partial_deriv_array);
        return fvar<T>(log_mix(theta, lambda1, lambda2.val_),
                       lambda2.d_ * value_of(partial_deriv_array[0]));
        }
    }
  }
}
#endif
