#ifndef STAN_MATH_REV_SCAL_FUN_CBRT_HPP
#define STAN_MATH_REV_SCAL_FUN_CBRT_HPP

#include <math.h>
#include <stan/math/rev/core.hpp>

namespace stan {
  namespace math {

    namespace {
      class cbrt_vari : public op_v_vari {
      public:
        explicit cbrt_vari(vari* avi) :
          op_v_vari(::cbrt(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (3.0 * val_ * val_);
        }
      };
    }

    /**
     * Returns the cube root of the specified variable (C99).
     *
     * See ::cbrt() for the double-based version.
     *
     * The derivative is
     *
     * \f$\frac{d}{dx} x^{1/3} = \frac{1}{3 x^{2/3}}\f$.
     *
       \f[
       \mbox{cbrt}(x) =
       \begin{cases}
         \sqrt[3]{x} & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{cbrt}(x)}{\partial x} =
       \begin{cases}
         \frac{1}{3x^{2/3}} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Specified variable.
     * @return Cube root of the variable.
     */
    inline var cbrt(const var& a) {
      return var(new cbrt_vari(a.vi_));
    }

  }
}
#endif
