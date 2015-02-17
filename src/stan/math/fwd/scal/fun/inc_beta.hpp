#ifndef STAN__MATH__FWD__SCAL__FUN__INC_BETA_HPP
#define STAN__MATH__FWD__SCAL__FUN__INC_BETA_HPP

#include <boost/math/special_functions/beta.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_beta.hpp>
#include <stan/math/fwd/scal/fun/grad_inc_beta.hpp>
#include <stan/math/rev/scal/fun/grad_inc_beta.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/lbeta.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/fwd/core/fvar.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline fvar<T> inc_beta(const fvar<T>& a,
                            const fvar<T>& b,
                            const fvar<T>& x) {
      using stan::math::digamma;
      using stan::math::grad_reg_inc_beta;
      using stan::math::inc_beta;
      using stan::math::lbeta;
      using stan::agrad::exp;
      using stan::agrad::digamma;
      using stan::agrad::lbeta;
      using stan::agrad::pow;
      using std::exp;
      using std::pow;

      T d_a; T d_b; T d_x;

      grad_reg_inc_beta(d_a,d_b,a.val_,b.val_,x.val_,
                    digamma(a.val_), digamma(b.val_),
                    digamma(a.val_+b.val_),
                    exp(lbeta(a.val_,b.val_)));
      d_x = pow((1-x.val_),b.val_-1)*pow(x.val_,a.val_-1)
        / exp(lbeta(a.val_, b.val_));
      return fvar<T>(inc_beta(a.val_, b.val_, x.val_),
                     a.d_ * d_a + b.d_ * d_b + x.d_ * d_x);
    }
  }
}

#endif
