#ifndef STAN__MATH__REV__SCAL__FUN__TANH_HPP
#define STAN__MATH__REV__SCAL__FUN__TANH_HPP

#include <cmath>
#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class tanh_vari : public op_v_vari {
      public:
        tanh_vari(vari* avi) :
          op_v_vari(std::tanh(avi->val_),avi) {
        }
        void chain() {
          double cosh = std::cosh(avi_->val_);
          avi_->adj_ += adj_ / (cosh * cosh);
        }
      };
    }

    /**
     * Return the hyperbolic tangent of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \tanh x = \frac{1}{\cosh^2 x}\f$.
     *
     *
       \f[
       \mbox{tanh}(x) =
       \begin{cases}
         \tanh(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{tanh}(x)}{\partial x} =
       \begin{cases}
         \mbox{sech}^2(x) & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Variable.
     * @return Hyperbolic tangent of variable.
     */
    inline var tanh(const var& a) {
      return var(new tanh_vari(a.vi_));
    }

  }
}
#endif
