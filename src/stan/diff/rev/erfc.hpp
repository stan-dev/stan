#ifndef __STAN__DIFF__REV__ERFC_HPP__
#define __STAN__DIFF__REV__ERFC_HPP__

#include <valarray>
#include <boost/math/special_functions/erf.hpp>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace diff {

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
    inline var erfc(const stan::diff::var& a) {
      return var(new erfc_vari(a.vi_));
    }

  }
}
#endif
