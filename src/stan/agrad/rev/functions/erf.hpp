#ifndef STAN__AGRAD__REV__FUNCTIONS__ERF_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ERF_HPP

#include <valarray>
#include <math.h>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class erf_vari : public op_v_vari {
      public:
        erf_vari(vari* avi) :
          op_v_vari(::erf(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * stan::math::TWO_OVER_SQRT_PI 
            * ::exp(- avi_->val_ * avi_->val_);
        }
      };
    }

    /**
     * The error function for variables (C99).
     *
     * For non-variable function, see ::erf() from math.h
     *
     * The derivative is
     *
     * \f$\frac{d}{dx} \mbox{erf}(x) = \frac{2}{\sqrt{\pi}} \exp(-x^2)\f$.
     * 
     *
       \f[
       \mbox{erf}(x) = 
       \begin{cases}
         \operatorname{erf}(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{erf}(x)}{\partial x} = 
       \begin{cases}
         \frac{\partial\, \operatorname{erf}(x)}{\partial x} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
   
       \f[
       \operatorname{erf}(x)=\frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}dt
       \f]
       
       \f[
       \frac{\partial \, \operatorname{erf}(x)}{\partial x} = \frac{2}{\sqrt{\pi}} e^{-x^2}
       \f]
     *
     * @param a The variable.
     * @return Error function applied to the variable.
     */
    inline var erf(const stan::agrad::var& a) {
      return var(new erf_vari(a.vi_));
    }

  }
}
#endif
