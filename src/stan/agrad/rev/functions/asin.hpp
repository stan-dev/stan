#ifndef STAN__AGRAD__REV__FUNCTIONS__ASIN_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ASIN_HPP

#include <cmath>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <math.h>

namespace stan {
  namespace agrad {
    
    namespace {
      class asin_vari : public op_v_vari {
      public:
        asin_vari(vari* avi) :
          op_v_vari(::asin(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / std::sqrt(1.0 - (avi_->val_ * avi_->val_));
        }
      };
    }
    
    /**
     * Return the principal value of the arc sine, in radians, of the
     * specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arcsin x = \frac{1}{\sqrt{1 - x^2}}\f$.
     *
     *
       \f[
       \mbox{asin}(x) = 
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < -1\\
         \arcsin(x) & \mbox{if } -1\leq x\leq 1 \\
         \textrm{NaN} & \mbox{if } x > 1\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{asin}(x)}{\partial x} = 
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < -1\\
         \frac{\partial\,\arcsin(x)}{\partial x} & \mbox{if } -1\leq x\leq 1 \\
         \textrm{NaN} & \mbox{if } x < -1\\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial \, \arcsin(x)}{\partial x} = \frac{1}{\sqrt{1-x^2}}
       \f]
     *
     * @param a Variable in range [-1,1].
     * @return Arc sine of variable, in radians. 
     */
    inline var asin(const var& a) {
      return var(new asin_vari(a.vi_));
    }

  }
}
#endif
