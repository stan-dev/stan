#ifndef __STAN__AGRAD__REV__ATANH_HPP__
#define __STAN__AGRAD__REV__ATANH_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <boost/math/special_functions/atanh.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class atanh_vari : public op_v_vari {
      public:
        atanh_vari(vari* avi) :
          op_v_vari(boost::math::atanh(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1.0 - avi_->val_ * avi_->val_);
        }
      };
    }

    /**
     * The inverse hyperbolic tangent function for variables (C99).
     *
     * For non-variable function, see boost::math::atanh().
     * 
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \mbox{atanh}(x) = \frac{1}{1 - x^2}\f$.
     *
     * @param a The variable.
     * @return Inverse hyperbolic tangent of the variable.
     */
    inline var atanh(const stan::agrad::var& a) {
      return var(new atanh_vari(a.vi_));
    }

  }
}
#endif
