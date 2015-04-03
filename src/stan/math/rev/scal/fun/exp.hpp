#ifndef STAN_MATH_REV_SCAL_FUN_EXP_HPP
#define STAN_MATH_REV_SCAL_FUN_EXP_HPP

#include <math.h>
#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class exp_vari : public op_v_vari {
      public:
        explicit exp_vari(vari* avi) :
          op_v_vari(::exp(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_;
        }
      };
    }

    /**
     * Return the exponentiation of the specified variable (cmath).
     *
       \f[
       \mbox{exp}(x) =
       \begin{cases}
         e^x & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{exp}(x)}{\partial x} =
       \begin{cases}
         e^x & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Variable to exponentiate.
     * @return Exponentiated variable.
     */
    inline var exp(const var& a) {
      return var(new exp_vari(a.vi_));
    }

  }
}
#endif
