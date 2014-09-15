#ifndef STAN__AGRAD__REV__FUNCTIONS__COS_HPP
#define STAN__AGRAD__REV__FUNCTIONS__COS_HPP

#include <math.h>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class cos_vari : public op_v_vari {
      public:
        cos_vari(vari* avi) :
          op_v_vari(::cos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ * ::sin(avi_->val_);
        }
      };
    }
    
    /**
     * Return the cosine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cos x = - \sin x\f$.
     *
     *
       \f[
       \mbox{cos}(x) = 
       \begin{cases}
         \cos(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{cos}(x)}{\partial x} = 
       \begin{cases}
         -\sin(x) & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Variable for radians of angle.
     * @return Cosine of variable. 
     */
    inline var cos(const var& a) {
      return var(new cos_vari(a.vi_));
    }

  }
}
#endif
