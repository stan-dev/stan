#ifndef STAN__AGRAD__REV__FUNCTIONS__EXPM1_HPP
#define STAN__AGRAD__REV__FUNCTIONS__EXPM1_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class expm1_vari : public op_v_vari {
      public:
        expm1_vari(vari* avi) :
          op_v_vari(std::exp(avi->val_) - 1.0,avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * (val_ + 1.0);
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
    inline var expm1(const stan::agrad::var& a) {
      return var(new expm1_vari(a.vi_));
    }

  }
}
#endif
