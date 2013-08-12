#ifndef __STAN__DIFF__REV__SINH_HPP__
#define __STAN__DIFF__REV__SINH_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {
    
    namespace {
      class sinh_vari : public op_v_vari {
      public:
        sinh_vari(vari* avi) :
          op_v_vari(std::sinh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::cosh(avi_->val_);
        }
      };
    }

    /**
     * Return the hyperbolic sine of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sinh x = \cosh x\f$.
     *
     * @param a Variable.
     * @return Hyperbolic sine of variable.
     */
    inline var sinh(const var& a) {
      return var(new sinh_vari(a.vi_));
    }
    
  }
}
#endif
