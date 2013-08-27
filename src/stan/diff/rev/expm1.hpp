#ifndef __STAN__DIFF__REV__EXPM1_HPP__
#define __STAN__DIFF__REV__EXPM1_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace diff {

    namespace {
      class expm1_vari : public op_v_vari {
      public:
        expm1_vari(vari* avi) :
          op_v_vari(std::exp(avi->val_) - 1.0,avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_;
        }
      };
    }

    /**
     * The exponentiation of the specified variable minus 1 (C99).
     *
     * For non-variable function, see boost::math::expm1().
     * 
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \exp(a) - 1 = \exp(a)\f$.
     * 
     * @param a The variable.
     * @return Two to the power of the specified variable.
     */
    inline var expm1(const stan::diff::var& a) {
      return var(new expm1_vari(a.vi_));
    }

  }
}
#endif
