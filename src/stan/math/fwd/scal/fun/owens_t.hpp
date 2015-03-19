#ifndef STAN__MATH__FWD__SCAL__FUN__OWENS_T_HPP
#define STAN__MATH__FWD__SCAL__FUN__OWENS_T_HPP

#include <math.h>
#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/owens_t.hpp>
#include <stan/math/prim/scal/fun/square.hpp>

namespace stan {
  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    owens_t(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::owens_t;
      using stan::math::pi;
      using stan::math::INV_SQRT_2;
      using stan::math::INV_SQRT_TWO_PI;
      using stan::math::square;
      using std::exp;
      using ::erf;

      T neg_x1_sq_div_2 = -square(x1.val_) * 0.5;
      T one_p_x2_sq = 1.0 + square(x2.val_);

      return fvar<T>(owens_t(x1.val_, x2.val_),
                     - x1.d_
                     * (erf(x2.val_ * x1.val_ * INV_SQRT_2)
                        * exp(neg_x1_sq_div_2) * INV_SQRT_TWO_PI * 0.5)
                     + x2.d_ * exp(neg_x1_sq_div_2 * one_p_x2_sq)
                     / (one_p_x2_sq * 2.0 * pi()));
    }

    template <typename T>
    inline
    fvar<T>
    owens_t(const double x1, const fvar<T>& x2) {
      using stan::math::owens_t;
      using stan::math::pi;
      using stan::math::square;
      using std::exp;

      T neg_x1_sq_div_2 = -square(x1) * 0.5;
      T one_p_x2_sq = 1.0 + square(x2.val_);

      return fvar<T>(owens_t(x1, x2.val_),
                     x2.d_ * exp(neg_x1_sq_div_2 * one_p_x2_sq)
                     / (one_p_x2_sq * 2.0 * pi()));
    }

    template <typename T>
    inline
    fvar<T>
    owens_t(const fvar<T>& x1, const double x2) {
      using stan::math::owens_t;
      using stan::math::pi;
      using stan::math::square;
      using stan::math::INV_SQRT_2;
      using stan::math::INV_SQRT_TWO_PI;
      using std::exp;
      using ::erf;

      T neg_x1_sq_div_2 = -square(x1.val_) * 0.5;

      return fvar<T>(owens_t(x1.val_, x2),
                     -x1.d_  * (erf(x2 * x1.val_ * INV_SQRT_2)
                                * exp(neg_x1_sq_div_2)
                                * INV_SQRT_TWO_PI * 0.5));
    }

  }
}
#endif
