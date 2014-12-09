#ifndef STAN__AGRAD__REV__FUNCTIONS__SIN_HPP
#define STAN__AGRAD__REV__FUNCTIONS__SIN_HPP

#include <cmath>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class sin_vari : public op_v_vari {
      public:
        sin_vari(vari* avi) :
          op_v_vari(std::sin(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::cos(avi_->val_);
        }
      };
    }

    /**
     * Return the sine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sin x = \cos x\f$.
     *
     *
       \f[
       \mbox{sin}(x) = 
       \begin{cases}
         \sin(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{sin}(x)}{\partial x} = 
       \begin{cases}
         \cos(x) & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Variable for radians of angle.
     * @return Sine of variable. 
     */
    inline var sin(const var& a) {
      return var(new sin_vari(a.vi_));
    }

  }
}
#endif
