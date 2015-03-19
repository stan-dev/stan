#ifndef STAN__MATH__REV__SCAL__FUN__ATAN_HPP
#define STAN__MATH__REV__SCAL__FUN__ATAN_HPP

#include <valarray>
#include <stan/math/rev/core.hpp>
#include <math.h>

namespace stan {
  namespace agrad {

    namespace {
      class atan_vari : public op_v_vari {
      public:
        atan_vari(vari* avi) :
          op_v_vari(::atan(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1.0 + (avi_->val_ * avi_->val_));
        }
      };
    }

    /**
     * Return the principal value of the arc tangent, in radians, of the
     * specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arctan x = \frac{1}{1 + x^2}\f$.
     *
     *
       \f[
       \mbox{atan}(x) =
       \begin{cases}
         \arctan(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{atan}(x)}{\partial x} =
       \begin{cases}
         \frac{\partial\, \arctan(x)}{\partial x} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial \, \arctan(x)}{\partial x} = \frac{1}{x^2+1}
       \f]
     *
     * @param a Variable in range [-1,1].
     * @return Arc tangent of variable, in radians.
     */
    inline var atan(const var& a) {
      return var(new atan_vari(a.vi_));
    }

  }
}
#endif
