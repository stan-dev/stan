#ifndef __STAN__PROB__INTERNAL_MATH__FWD__INC_BETA_HPP__
#define __STAN__PROB__INTERNAL_MATH__FWD__INC_BETA_HPP__

#include <boost/math/special_functions/beta.hpp>
#include <stan/prob/internal_math/math/grad_inc_beta.hpp>
#include <stan/agrad/fwd/functions/pow.hpp>
#include <stan/agrad/rev/functions/pow.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline fvar<T> inc_beta(const fvar<T>& a,
                        const fvar<T>& b,
                        const fvar<T>& x) {
      using boost::math::beta;
      using stan::agrad::pow;
      using std::pow;

      T d_a; T d_b; T d_x;
      stan::math::gradIncinc_beta(d_a,d_b,a.val_,b.val_,x.val_);
      d_x = pow((1-x.val_),b.val_-1)*pow(x.val_,a.val_-1);
      return fvar<T>(inc_beta(a.val_, b.val_, x.val_),
                     a.d_ * d_a + b.d_ * d_b + x.d_ * d_x);
    }
    template<typename T>
    inline fvar<T> inc_beta(const fvar<T>& a,
                        const fvar<T>& b,
                        const double& x) {
      using boost::math::beta;
      using stan::agrad::pow;
      using std::pow;

      T d_a; T d_b;
      T x_val(x);
      stan::math::gradIncinc_beta(d_a,d_b,a.val_,b.val_,x_val);
      return fvar<T>(inc_beta(a.val_, b.val_, x),
                     a.d_ * d_a + b.d_ * d_b);
    }
    template<typename T>
    inline fvar<T> inc_beta(const fvar<T>& a,
                        const double& b,
                        const fvar<T>& x) {
      using boost::math::beta;
      using stan::agrad::pow;
      using std::pow;

      T d_a; T d_b; T d_x; T b_val(b);
      stan::math::gradIncinc_beta(d_a,d_b,a.val_,b_val,x.val_);
      d_x = pow((1-x.val_),b-1)*pow(x.val_,a.val_-1);
      return fvar<T>(inc_beta(a.val_, b, x.val_),
                     a.d_ * d_a + x.d_ * d_x);
    }
    template<typename T>
    inline fvar<T> inc_beta(const double& a,
                        const fvar<T>& b,
                        const fvar<T>& x) {
      using boost::math::beta;
      using stan::agrad::pow;
      using std::pow;

      T d_a; T d_b; T d_x; T a_val(a);
      stan::math::gradIncinc_beta(d_a,d_b,a_val,b.val_,x.val_);
      d_x = pow((1-x.val_),b.val_-1)*pow(x.val_,a-1);
      return fvar<T>(inc_beta(a, b.val_, x.val_),
                     b.d_ * d_b + x.d_ * d_x);
    }
    template<typename T>
    inline fvar<T> inc_beta(const double& a,
                        const double& b,
                        const fvar<T>& x) {
      using boost::math::beta;
      using stan::agrad::pow;
      using std::pow;

      T d_x;
      d_x = pow((1-x.val_),b-1)*pow(x.val_,a-1);
      return fvar<T>(inc_beta(a, b, x.val_),
                     x.d_ * d_x);
    }
    template<typename T>
    inline fvar<T> inc_beta(const double& a,
                        const fvar<T>& b,
                        const double& x) {
      using boost::math::beta;
      using stan::agrad::pow;
      using std::pow;

      T d_a; T d_b; T d_x; T x_val(x); T a_val(a);
      stan::math::gradIncinc_beta(d_a,d_b,a_val,b.val_,x_val);
      return fvar<T>(inc_beta(a, b.val_, x),
                     b.d_ * d_b);
    }
    template<typename T>
    inline fvar<T> inc_beta(const fvar<T>& a,
                        const double& b,
                        const double& x) {
      using boost::math::beta;
      using stan::agrad::pow;
      using std::pow;

      T d_a; T d_b; T d_x; T b_val(b); T x_val(x);
      stan::math::gradIncinc_beta(d_a,d_b,a.val_,b_val,x_val);
      return fvar<T>(inc_beta(a.val_, b, x),
                     a.d_ * d_a);
    }
  }
}

#endif
