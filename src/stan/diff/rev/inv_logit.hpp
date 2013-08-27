#ifndef __STAN__DIFF__REV__INV_LOGIT_HPP__
#define __STAN__DIFF__REV__INV_LOGIT_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/math/functions/inv_logit.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {

    namespace {
      class inv_logit_vari : public op_v_vari {
      public:
        inv_logit_vari(vari* avi) :
        op_v_vari(stan::math::inv_logit(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ +=  adj_ * val_ * (1.0 - val_);
        }
      };
    }

    /**
     * The inverse logit function for variables (stan).
     *
     * See stan::math::inv_logit() for the double-based version.
     *
     * The derivative of inverse logit is
     *
     * \f$\frac{d}{dx} \mbox{logit}^{-1}(x) = \mbox{logit}^{-1}(x) (1 - \mbox{logit}^{-1}(x))\f$.
     *
     * @param a Argument variable.
     * @return Inverse logit of argument.
     */
    inline var inv_logit(const stan::diff::var& a) {
      return var(new inv_logit_vari(a.vi_));
    }

  }
}
#endif
