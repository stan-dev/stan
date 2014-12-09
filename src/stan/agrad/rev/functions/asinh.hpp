#ifndef STAN__AGRAD__REV__FUNCTIONS__ASINH_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ASINH_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>
#include <stan/agrad/rev/operators/operator_equal.hpp>
#include <stan/agrad/rev/operators/operator_unary_negative.hpp>
#include <math.h>

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
     *
       \f[
       \mbox{asinh}(x) = 
       \begin{cases}
         \sinh^{-1}(x) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{asinh}(x)}{\partial x} = 
       \begin{cases}
         \frac{\partial\, \sinh^{-1}(x)}{\partial x} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
   
       \f[
       \sinh^{-1}(x)=\ln\left(x+\sqrt{x^2+1}\right)
       \f]
       
       \f[
       \frac{\partial \, \sinh^{-1}(x)}{\partial x} = \frac{1}{\sqrt{x^2+1}}
       \f]
     *
     * @param a The variable.
     * @return Inverse hyperbolic sine of the variable.
     */
    inline var asinh(const stan::agrad::var& a) {
      if (boost::math::isinf(a.val()))
        return var(new asinh_vari(a.val(), a.vi_));
      return var(new asinh_vari(::asinh(a.val()),a.vi_));
    }

  }
}
#endif
