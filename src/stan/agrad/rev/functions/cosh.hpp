#ifndef STAN__AGRAD__REV__FUNCTIONS__COSH_HPP
#define STAN__AGRAD__REV__FUNCTIONS__COSH_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>

namespace stan {
  namespace agrad {
    
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
