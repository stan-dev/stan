#ifndef __STAN__AGRAD__REV__ATANH_HPP__
#define __STAN__AGRAD__REV__ATANH_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <boost/math/special_functions/atanh.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>
#include <stan/math/error_handling/check_less.hpp>
#include <stan/math/error_handling/check_greater.hpp>
#include <limits>
#include <stan/agrad/rev/operator_equal.hpp>

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
      static const char* function = "stan::agrad::acos(%1%)";
      if (a == 1.0)
	return var(new atanh_vari(std::numeric_limits<double>::infinity(),a.vi_));
      if (a == -1.0)
	return var(new atanh_vari(-std::numeric_limits<double>::infinity(),a.vi_));
      if (!stan::math::check_greater(function,a.val(),-1.0,"angle")
	  && !stan::math::check_less(function,a.val(), 1.0,"angle"))
	return std::numeric_limits<double>::quiet_NaN();
      return var(new atanh_vari(boost::math::atanh(a.val()),a.vi_));
    }

  }
}
#endif
