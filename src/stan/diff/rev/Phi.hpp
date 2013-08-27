#ifndef __STAN__DIFF__REV__PHI_HPP__
#define __STAN__DIFF__REV__PHI_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/math/functions/Phi.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {

    namespace {
      class Phi_vari : public op_v_vari {
      public:
        Phi_vari(vari* avi) :
          op_v_vari(stan::math::Phi(avi->val_), avi) {
        }
        void chain() {
          static const double NEG_HALF = -0.5;
          avi_->adj_ += adj_ 
            * stan::math::INV_SQRT_TWO_PI * std::exp(NEG_HALF * avi_->val_ * avi_->val_);
        }
      };
    }

    /**
     * The unit normal cumulative density function for variables (stan).
     *
     * See stan::math::Phi() for the double-based version.
     *
     * The derivative is the unit normal density function,
     *
     * \f$\frac{d}{dx} \Phi(x) = \mbox{\sf Norm}(x|0,1) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{1}{2} x^2)\f$.
     *
     * @param a Variable argument.
     * @return The unit normal cdf evaluated at the specified argument.
     */
    inline var Phi(const stan::diff::var& a) {
      return var(new Phi_vari(a.vi_));
    }

  }
}
#endif
