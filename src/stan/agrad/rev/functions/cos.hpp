#ifndef STAN__AGRAD__REV__FUNCTIONS__COS_HPP
#define STAN__AGRAD__REV__FUNCTIONS__COS_HPP

#include <cmath>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class cos_vari : public op_v_vari {
      public:
        cos_vari(vari* avi) :
          op_v_vari(std::cos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ * std::sin(avi_->val_);
        }
      };
    }
    
    /**
     * Return the cosine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cos x = - \sin x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Cosine of variable. 
     */
    inline var cos(const var& a) {
      return var(new cos_vari(a.vi_));
    }

  }
}
#endif
