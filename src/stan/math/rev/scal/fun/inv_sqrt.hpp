#ifndef STAN__MATH__REV__SCAL__FUN__INV_SQRT_HPP
#define STAN__MATH__REV__SCAL__FUN__INV_SQRT_HPP

#include <valarray>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/v_vari.hpp>
#include <stan/math/prim/scal/fun/inv_sqrt.hpp>

namespace stan {
  namespace agrad {
    
    namespace {
      class inv_sqrt_vari : public op_v_vari {
      public:
        inv_sqrt_vari(vari* avi) :
          op_v_vari(stan::math::inv_sqrt(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= 0.5 * adj_ / (avi_->val_ * std::sqrt(avi_->val_));
        }
      };
    }
    
    /**
     *
       \f[
       \mbox{inv\_sqrt}(x) = 
       \begin{cases}
         \frac{1}{\sqrt{x}} & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{inv\_sqrt}(x)}{\partial x} = 
       \begin{cases}
         -\frac{1}{2\sqrt{x^3}} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     */
    inline var inv_sqrt(const var& a) {
      return var(new inv_sqrt_vari(a.vi_));
    }

  }
}
#endif
