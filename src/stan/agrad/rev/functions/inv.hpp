#ifndef STAN__AGRAD__REV__FUNCTIONS__INV_HPP
#define STAN__AGRAD__REV__FUNCTIONS__INV_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/functions/inv.hpp>

namespace stan {
  namespace agrad {
    
    namespace {
      class inv_vari : public op_v_vari {
      public:
        inv_vari(vari* avi) :
          op_v_vari(stan::math::inv(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ / (avi_->val_ * avi_->val_);
        }
      };
    }

    /**
     *
       \f[
       \mbox{inv}(x) = 
       \begin{cases}
         \frac{1}{x} & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{inv}(x)}{\partial x} = 
       \begin{cases}
         -\frac{1}{x^2} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     */
    inline var inv(const var& a) {
      return var(new inv_vari(a.vi_));
    }

  }
}
#endif
