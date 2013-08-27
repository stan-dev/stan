#ifndef __STAN__AGRAD__REV__TAN_HPP__
#define __STAN__AGRAD__REV__TAN_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <limits>
#include <stan/math/error_handling/check_finite.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class tan_vari : public op_v_vari {
      public:
        tan_vari(vari* avi) :
          op_v_vari(std::tan(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * (1.0 + val_ * val_); 
        }
      };
    }
    
    /**
     * Return the tangent of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \tan x = \sec^2 x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Tangent of variable. 
     */
    inline var tan(const var& a) {
      static const char* function = "stan::agrad::tan(%1%)";
      if (!stan::math::check_finite(function,a.val(),"angle"))
        return std::numeric_limits<double>::quiet_NaN();
      return var(new tan_vari(a.vi_));
    }

  }
}
#endif
