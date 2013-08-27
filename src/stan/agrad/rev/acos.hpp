#ifndef __STAN__AGRAD__REV__ACOS_HPP__
#define __STAN__AGRAD__REV__ACOS_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <limits>
#include <stan/math/error_handling/check_bounded.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class acos_vari : public op_v_vari {
      public:
        acos_vari(vari* avi) :
          op_v_vari(std::acos(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ / std::sqrt(1.0 - (avi_->val_ * avi_->val_));
        }
      };
    }

    /**
     * Return the principal value of the arc cosine of a variable,
     * in radians (cmath).
     *
     * The derivative is defined by
     *
     * \f$\frac{d}{dx} \arccos x = \frac{-1}{\sqrt{1 - x^2}}\f$.
     *
     * @param a Variable in range [-1,1].
     * @return Arc cosine of variable, in radians. 
     */
    inline var acos(const var& a) {
      static const char* function = "stan::agrad::acos(%1%)";
      if (!stan::math::check_bounded(function,a.val(),-1.0,1.0,"angle"))
        return std::numeric_limits<double>::quiet_NaN();
      return var(new acos_vari(a.vi_));
    }

  }
}
#endif
