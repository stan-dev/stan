#ifndef STAN__AGRAD__REV__FUNCTIONS__ERFC_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ERFC_HPP

#include <valarray>
#include <boost/math/special_functions/erf.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class erfc_vari : public op_v_vari {
      public:
        erfc_vari(vari* avi) :
          op_v_vari(boost::math::erfc(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * stan::math::NEG_TWO_OVER_SQRT_PI * std::exp(- avi_->val_ * avi_->val_);
        }
      };
    }

    /**
     * The complementary error function for variables (C99).
     *
     * For non-variable function, see boost::math::erfc().
     *
     * The derivative is
     * 
     * \f$\frac{d}{dx} \mbox{erfc}(x) = - \frac{2}{\sqrt{\pi}} \exp(-x^2)\f$.
     *
     * @param a The variable.
     * @return Complementary error function applied to the variable.
     */
    inline var erfc(const stan::agrad::var& a) {
      return var(new erfc_vari(a.vi_));
    }

  }
}
#endif
