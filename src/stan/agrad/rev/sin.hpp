#ifndef __STAN__AGRAD__REV__SIN_HPP__
#define __STAN__AGRAD__REV__SIN_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <limits>
#include <stan/math/error_handling/check_finite.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class sin_vari : public op_v_vari {
      public:
        sin_vari(vari* avi) :
          op_v_vari(std::sin(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::cos(avi_->val_);
        }
      };
    }

    /**
     * Return the sine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \sin x = \cos x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Sine of variable. 
     */
    inline var sin(const var& a) {
      static const char* function = "stan::agrad::sin(%1%)";
      if (!stan::math::check_finite(function,a.val(),"angle"))
          return std::numeric_limits<double>::quiet_NaN();
      return var(new sin_vari(a.vi_));
    }

  }
}
#endif
