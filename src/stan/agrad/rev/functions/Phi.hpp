#ifndef STAN__AGRAD__REV__FUNCTIONS__PHI_HPP
#define STAN__AGRAD__REV__FUNCTIONS__PHI_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/math/functions/Phi.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>

namespace stan {
  namespace agrad {

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
    inline var Phi(const stan::agrad::var& a) {
      return var(new Phi_vari(a.vi_));
    }

  }
}
#endif
