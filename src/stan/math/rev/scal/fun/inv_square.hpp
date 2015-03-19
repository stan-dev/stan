#ifndef STAN__MATH__REV__SCAL__FUN__INV_SQUARE_HPP
#define STAN__MATH__REV__SCAL__FUN__INV_SQUARE_HPP

#include <valarray>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/inv_square.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class inv_square_vari : public op_v_vari {
      public:
        inv_square_vari(vari* avi) :
          op_v_vari(stan::math::inv_square(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= 2 * adj_ / (avi_->val_ * avi_->val_ * avi_->val_);
        }
      };
    }

    /**
     *
       \f[
       \mbox{inv\_square}(x) =
       \begin{cases}
         \frac{1}{x^2} & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{inv\_square}(x)}{\partial x} =
       \begin{cases}
         -\frac{2}{x^3} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     */
    inline var inv_square(const var& a) {
      return var(new inv_square_vari(a.vi_));
    }

  }
}
#endif
