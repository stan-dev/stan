#ifndef STAN__AGRAD__REV__FUNCTIONS__ACOS_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ACOS_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <math.h>

namespace stan {
  namespace agrad {

    namespace {
      class acos_vari : public op_v_vari {
      public:
        acos_vari(vari* avi) :
          op_v_vari(::acos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ / std::sqrt(1.0 - (avi_->val_ * avi_->val_));
        }
      };
    }

    /**
     * Return the principal value of the arc cosine of a variable,
     * in radians (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arccos x = \frac{-1}{\sqrt{1 - x^2}}\f$.
     *
     *
       \f[
       \mbox{acos}(x) = 
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < -1\\
         \arccos(x) & \mbox{if } -1\leq x\leq 1 \\
         \textrm{NaN} & \mbox{if } x > 1\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{acos}(x)}{\partial x} = 
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < -1\\
         \frac{\partial\,\arccos(x)}{\partial x} & \mbox{if } -1\leq x\leq 1 \\
         \textrm{NaN} & \mbox{if } x < -1\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial \, \arccos(x)}{\partial x} = -\frac{1}{\sqrt{1-x^2}}
       \f]
     *
     * @param a Variable in range [-1,1].
     * @return Arc cosine of variable, in radians. 
     */
    inline var acos(const var& a) {
      return var(new acos_vari(a.vi_));
    }

  }
}
#endif
