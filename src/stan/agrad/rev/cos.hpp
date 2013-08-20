#ifndef __STAN__AGRAD__REV__COS_HPP__
#define __STAN__AGRAD__REV__COS_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <limits>
#include <stan/math/error_handling/check_finite.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class cos_vari : public op_v_vari {
      public:
        cos_vari(vari* avi) :
          op_v_vari(std::cos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ * std::sin(avi_->val_);
        }
      };
    }
    
    /**
     * Return the cosine of a radian-scaled variable (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \cos x = - \sin x\f$.
     *
     * @param a Variable for radians of angle.
     * @return Cosine of variable. 
     */
    inline var cos(const var& a) {
static const char* function = "stan::agrad::cos(%1%)";
 if (!stan::math::check_finite(function,a.val(),"angle"))
   return std::numeric_limits<double>::quiet_NaN();
 return var(new cos_vari(a.vi_));
    }

  }
}
#endif
