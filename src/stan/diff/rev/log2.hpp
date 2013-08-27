#ifndef __STAN__DIFF__REV__LOG2_HPP__
#define __STAN__DIFF__REV__LOG2_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/functions/log2.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace diff {

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
     * @param a Specified variable.
     * @return Base 2 logarithm of the variable.
     */
    inline var log2(const stan::diff::var& a) {
      return var(new log2_vari(a.vi_));
    }

  }
}
#endif
