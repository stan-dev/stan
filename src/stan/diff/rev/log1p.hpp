#ifndef __STAN__DIFF__REV__LOG1P_HPP__
#define __STAN__DIFF__REV__LOG1P_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/log1p.hpp>

namespace stan {
  namespace diff {

    namespace {
      class log1p_vari : public op_v_vari {
      public:
        log1p_vari(vari* avi) :
          op_v_vari(stan::math::log1p(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1 + avi_->val_);
        }
      };
    }

    /**
     * The log (1 + x) function for variables (C99).
     *
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \log (1 + x) = \frac{1}{1 + x}\f$.
     *
     * @param a The variable.
     * @return The log of 1 plus the variable.
     */
    inline var log1p(const stan::diff::var& a) {
      return var(new log1p_vari(a.vi_));
    }

  }
}
#endif
