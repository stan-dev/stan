#ifndef __STAN__DIFF__REV__LOG10_HPP__
#define __STAN__DIFF__REV__LOG10_HPP__

#include <cmath>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace diff {

    namespace {
      class log10_vari : public op_v_vari {
      public:
        const double exp_val_;
        log10_vari(vari* avi) :
        op_v_vari(std::log10(avi->val_),avi),
        exp_val_(avi->val_) {
        }
        void chain() {
          avi_->adj_ += adj_ / (stan::math::LOG_10 * exp_val_);
        }
      };
    }

    /**
     * Return the base 10 log of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \log_{10} x = \frac{1}{x \log 10}\f$.
     * 
     * @param a Variable whose log is taken.
     * @return Base 10 log of variable.
     */
    inline var log10(const var& a) {
      return var(new log10_vari(a.vi_));
    }

  }
}
#endif
