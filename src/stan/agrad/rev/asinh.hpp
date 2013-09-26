#ifndef __STAN__AGRAD__REV__ASINH_HPP__
#define __STAN__AGRAD__REV__ASINH_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <boost/math/special_functions/asinh.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class asinh_vari : public op_v_vari {
      public:
        asinh_vari(double val, vari* avi) :
          op_v_vari(val,avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / std::sqrt(avi_->val_ * avi_->val_ + 1.0);
        }
      };
    }

    /**
     * The inverse hyperbolic sine function for variables (C99).
     * 
     * For non-variable function, see boost::math::asinh().
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \mbox{asinh}(x) = \frac{x}{x^2 + 1}\f$.
     *
     * @param a The variable.
     * @return Inverse hyperbolic sine of the variable.
     */
    inline var asinh(const stan::agrad::var& a) {
      if (std::isinf(a))
        return var(new asinh_vari(a.val(), a.vi_));
      return var(new asinh_vari(boost::math::asinh(a.val()),a.vi_));
    }

  }
}
#endif
