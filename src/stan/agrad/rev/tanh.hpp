#ifndef __STAN__AGRAD__REV__TANH_HPP__
#define __STAN__AGRAD__REV__TANH_HPP__

#include <cmath>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class tanh_vari : public op_v_vari {
      public:
        tanh_vari(vari* avi) :
          op_v_vari(std::tanh(avi->val_),avi) {
        }
        void chain() {
          double cosh = std::cosh(avi_->val_);
          avi_->adj_ += adj_ / (cosh * cosh);
        }
      };
    }

    /**
     * Return the hyperbolic tangent of the specified variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \tanh x = \frac{1}{\cosh^2 x}\f$.
     *
     * @param a Variable.
     * @return Hyperbolic tangent of variable.
     */
    inline var tanh(const var& a) {
      return var(new tanh_vari(a.vi_));
    }

  }
}
#endif
