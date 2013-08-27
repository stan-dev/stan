#ifndef __STAN__DIFF__REV__COSH_HPP__
#define __STAN__DIFF__REV__COSH_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {
    
    namespace {
      class cosh_vari : public op_v_vari {
      public:
        cosh_vari(vari* avi) :
          op_v_vari(std::cosh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::sinh(avi_->val_);
        }
      };
    }
    
    /**
     * Return the hyperbolic cosine of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cosh x = \sinh x\f$.
     *
     * @param a Variable.
     * @return Hyperbolic cosine of variable.
     */
    inline var cosh(const var& a) {
      return var(new cosh_vari(a.vi_));
    }
    
  }
}
#endif
