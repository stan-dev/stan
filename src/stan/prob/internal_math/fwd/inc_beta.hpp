#ifndef STAN__PROB__INTERNAL_MATH__FWD__INC_BETA_HPP
#define STAN__PROB__INTERNAL_MATH__FWD__INC_BETA_HPP

#include <boost/math/special_functions/beta.hpp>
#include <stan/prob/internal_math/math/grad_inc_beta.hpp>
#include <stan/prob/internal_math/fwd/grad_inc_beta.hpp>
#include <stan/prob/internal_math/rev/grad_inc_beta.hpp>
#include <stan/prob/internal_math/rev/inc_beta.hpp>
#include <stan/agrad/fwd/functions/pow.hpp>
#include <stan/agrad/rev/functions/pow.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline fvar<T> inc_beta(const fvar<T>& a,
                            const fvar<T>& b,
                            const fvar<T>& x) {
      using stan::math::grad_inc_beta;
      using stan::agrad::grad_inc_beta;
      using stan::math::inc_beta;
      using stan::agrad::pow;
      using std::pow;
      T d_a; T d_b; T d_x;

      grad_inc_beta(d_a,d_b,a.val_,b.val_,x.val_);
      d_x = pow((1-x.val_),b.val_-1)*pow(x.val_,a.val_-1);
      return fvar<T>(inc_beta(a.val_, b.val_, x.val_),
                     a.d_ * d_a + b.d_ * d_b + x.d_ * d_x);
    }
  }
}

#endif
