#ifndef STAN__AGRAD__REV__FUNCTIONS__ATANH_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ATANH_HPP

#include <boost/math/special_functions/atanh.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/agrad/rev/operators/operator_equal.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class atanh_vari : public op_v_vari {
      public:
        atanh_vari(double val, vari* avi) :
          op_v_vari(val,avi) {
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
      if (a == 1.0)
        return var(new atanh_vari(std::numeric_limits<double>::infinity(),a.vi_));
      if (a == -1.0)
        return var(new atanh_vari(-std::numeric_limits<double>::infinity(),a.vi_));
      return var(new atanh_vari(boost::math::atanh(a.val()),a.vi_));
    }

  }
}
#endif
