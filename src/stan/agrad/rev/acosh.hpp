#ifndef __STAN__AGRAD__REV__ACOSH_HPP__
#define __STAN__AGRAD__REV__ACOSH_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <boost/math/special_functions/acosh.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class acosh_vari : public op_v_vari {
      public:
        acosh_vari(vari* avi) :
          op_v_vari(boost::math::acosh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / std::sqrt(avi_->val_ * avi_->val_ - 1.0);
        }
      };
    }

    /**
     * The inverse hyperbolic cosine function for variables (C99).
     * 
     * For non-variable function, see boost::math::acosh().
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \mbox{acosh}(x) = \frac{x}{x^2 - 1}\f$.
     *
     * @param a The variable.
     * @return Inverse hyperbolic cosine of the variable.
     */
    inline var acosh(const stan::agrad::var& a) {
      return var(new acosh_vari(a.vi_));
    }

  }
}
#endif
