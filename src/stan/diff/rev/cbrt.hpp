#ifndef __STAN__DIFF__REV__CBRT_HPP__
#define __STAN__DIFF__REV__CBRT_HPP__

#include <boost/math/special_functions/cbrt.hpp>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {

    namespace {
      class cbrt_vari : public op_v_vari {
      public:
        cbrt_vari(vari* avi) :
          op_v_vari(boost::math::cbrt(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (3.0 * val_ * val_);
        }
      };
    }

    /**
     * Returns the cube root of the specified variable (C99).
     *
     * See boost::math::cbrt() for the double-based version.
     *
     * The derivative is
     *
     * \f$\frac{d}{dx} x^{1/3} = \frac{1}{3 x^{2/3}}\f$.
     *
     * @param a Specified variable.
     * @return Cube root of the variable.
     */
    inline var cbrt(const stan::diff::var& a) {
      return var(new cbrt_vari(a.vi_));
    }

  }
}
#endif
