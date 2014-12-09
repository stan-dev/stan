#ifndef STAN__AGRAD__REV__FUNCTIONS__COSH_HPP
#define STAN__AGRAD__REV__FUNCTIONS__COSH_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <math.h>

namespace stan {
  namespace agrad {
    
    namespace {
      class cosh_vari : public op_v_vari {
      public:
        cosh_vari(vari* avi) :
          op_v_vari(::cosh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * ::sinh(avi_->val_);
        }
      };
    }
    
    /**
     * Return the hyperbolic cosine of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cosh x = \sinh x\f$.
     *
     *
       \f[
       \mbox{cosh}(x) = 
       \begin{cases}
         \cosh(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{cosh}(x)}{\partial x} = 
       \begin{cases}
         \sinh(x) & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Variable.
     * @return Hyperbolic cosine of variable.
     */
    inline var cosh(const var& a) {
      return var(new cosh_vari(a.vi_));
    }
    
  }
}
#endif
