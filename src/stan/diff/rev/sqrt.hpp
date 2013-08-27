#ifndef __STAN__DIFF__REV__SQRT_HPP__
#define __STAN__DIFF__REV__SQRT_HPP__

#include <cmath>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {
    
    namespace {
      class sqrt_vari : public op_v_vari {
      public:
        sqrt_vari(vari* avi) :
          op_v_vari(std::sqrt(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (2.0 * val_);
        }
      };
    }

    /**
     * Return the square root of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sqrt{x} = \frac{1}{2 \sqrt{x}}\f$.
     * 
     * @param a Variable whose square root is taken.
     * @return Square root of variable.
     */
    inline var sqrt(const var& a) {
      return var(new sqrt_vari(a.vi_));
    }

  }
}
#endif
