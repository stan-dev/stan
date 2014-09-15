#ifndef STAN__AGRAD__REV__FUNCTIONS__LOG2_HPP
#define STAN__AGRAD__REV__FUNCTIONS__LOG2_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/functions/log2.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class log2_vari : public op_v_vari {
      public:
        log2_vari(vari* avi) :
          op_v_vari(stan::math::log2(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (stan::math::LOG_2 * avi_->val_); 
        }
      };
    }

    /**
     * Returns the base 2 logarithm of the specified variable (C99).
     *
     * See stan::math::log2() for the double-based version.
     *
     * The derivative is
     *
     * \f$\frac{d}{dx} \log_2 x = \frac{1}{x \log 2}\f$.
     *
       \f[
       \mbox{log2}(x) = 
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < 0 \\
         \log_2(x) & \mbox{if } x\geq 0 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{log2}(x)}{\partial x} = 
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < 0 \\
         \frac{1}{x\ln2} & \mbox{if } x\geq 0 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Specified variable.
     * @return Base 2 logarithm of the variable.
     */
    inline var log2(const stan::agrad::var& a) {
      return var(new log2_vari(a.vi_));
    }

  }
}
#endif
