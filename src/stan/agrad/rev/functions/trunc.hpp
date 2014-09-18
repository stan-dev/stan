#ifndef STAN__AGRAD__REV__FUNCTIONS__TRUNC_HPP
#define STAN__AGRAD__REV__FUNCTIONS__TRUNC_HPP

#include <math.h>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class trunc_vari : public op_v_vari {
      public:
        trunc_vari(vari* avi) :
          op_v_vari(::trunc(avi->val_),avi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
        }
      };
    }

    /**
     * Returns the truncatation of the specified variable (C99).
     *
     * See boost::math::trunc() for the double-based version.
     *
     * The derivative is zero everywhere but at integer values, so for
     * convenience the derivative is defined to be everywhere zero,
     *
     * \f$\frac{d}{dx} \mbox{trunc}(x) = 0\f$.
     *
     *
       \f[
       \mbox{trunc}(x) = 
       \begin{cases}
         \lfloor x \rfloor & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{trunc}(x)}{\partial x} = 
       \begin{cases}
         0 & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Specified variable.
     * @return Truncation of the variable.
     */
    inline var trunc(const stan::agrad::var& a) {
      return var(new trunc_vari(a.vi_));
    }

  }
}
#endif
