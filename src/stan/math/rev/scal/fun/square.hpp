#ifndef STAN__MATH__REV__SCAL__FUN__SQUARE_HPP
#define STAN__MATH__REV__SCAL__FUN__SQUARE_HPP

#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class square_vari : public op_v_vari {
      public:
        square_vari(vari* avi) :
          op_v_vari(avi->val_ * avi->val_,avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * 2.0 * avi_->val_;
        }
      };
    }

    /**
     * Return the square of the input variable.
     *
     * <p>Using <code>square(x)</code> is more efficient
     * than using <code>x * x</code>.
     *
       \f[
       \mbox{square}(x) =
       \begin{cases}
         x^2 & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{square}(x)}{\partial x} =
       \begin{cases}
         2x & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param x Variable to square.
     * @return Square of variable.
     */
    inline var square(const var& x) {
      return var(new square_vari(x.vi_));
    }

  }
}
#endif
