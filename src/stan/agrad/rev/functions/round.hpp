#ifndef STAN__AGRAD__REV__FUNCTIONS__ROUND_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ROUND_HPP

#include <math.h>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class round_vari : public op_v_vari {
      public:
        round_vari(vari* avi) :
          op_v_vari(::round(avi->val_),avi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
        }
      };
    }

    /**
     * Returns the rounded form of the specified variable (C99).
     *
     * See boost::math::round() for the double-based version.
     *
     * The derivative is zero everywhere but numbers half way between
     * whole numbers, so for convenience the derivative is defined to
     * be everywhere zero,
     *
     * \f$\frac{d}{dx} \mbox{round}(x) = 0\f$.
     *
     *
       \f[
       \mbox{round}(x) = 
       \begin{cases}
         \lceil x \rceil & \mbox{if } x-\lfloor x\rfloor \geq 0.5 \\
         \lfloor x \rfloor & \mbox{if } x-\lfloor x\rfloor < 0.5 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{round}(x)}{\partial x} = 
       \begin{cases}
         0 & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Specified variable.
     * @return Rounded variable.
     */
    inline var round(const stan::agrad::var& a) {
      return var(new round_vari(a.vi_));
    }

  }
}
#endif
