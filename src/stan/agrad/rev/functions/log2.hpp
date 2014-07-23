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
     * @param a Specified variable.
     * @return Base 2 logarithm of the variable.
     */
    inline var log2(const stan::agrad::var& a) {
      return var(new log2_vari(a.vi_));
    }

  }
}
#endif
