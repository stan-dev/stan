#ifndef STAN__MATH__REV__SCAL__FUN__ACOSH_HPP
#define STAN__MATH__REV__SCAL__FUN__ACOSH_HPP

#include <math.h>
#include <stan/math/rev/core.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class acosh_vari : public op_v_vari {
      public:
        acosh_vari(double val, vari* avi) :
          op_v_vari(val,avi) {
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
     *
       \f[
       \mbox{acosh}(x) =
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < 1 \\
         \cosh^{-1}(x) & \mbox{if } x \geq 1 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{acosh}(x)}{\partial x} =
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < 1 \\
         \frac{\partial\, \cosh^{-1}(x)}{\partial x} & \mbox{if } x \geq 1 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \cosh^{-1}(x)=\ln\left(x+\sqrt{x^2-1}\right)
       \f]

       \f[
       \frac{\partial \, \cosh^{-1}(x)}{\partial x} = \frac{1}{\sqrt{x^2-1}}
       \f]
     *
     * @param a The variable.
     * @return Inverse hyperbolic cosine of the variable.
     */
    inline var acosh(const stan::agrad::var& a) {
      if (boost::math::isinf(a.val()) && a > 0.0)
        return var(new acosh_vari(a.val(),a.vi_));
      return var(new acosh_vari(::acosh(a.val()),a.vi_));
    }

  }
}

#endif
