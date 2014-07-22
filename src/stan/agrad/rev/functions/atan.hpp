#ifndef STAN__AGRAD__REV__FUNCTIONS__ATAN_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ATAN_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>

namespace stan {
  namespace agrad {
    
    namespace {
      class atan_vari : public op_v_vari {
      public:
        atan_vari(vari* avi) :
          op_v_vari(std::atan(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1.0 + (avi_->val_ * avi_->val_));
        }
      };
    }

    /**
     * Return the principal value of the arc tangent, in radians, of the
     * specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arctan x = \frac{1}{1 + x^2}\f$.
     *
     * @param a Variable in range [-1,1].
     * @return Arc tangent of variable, in radians. 
     */
    inline var atan(const var& a) {
      return var(new atan_vari(a.vi_));
    }

  }
}
#endif
