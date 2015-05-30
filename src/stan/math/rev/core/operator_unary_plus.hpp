#ifndef STAN_MATH_REV_CORE_OPERATOR_UNARY_PLUS_HPP
#define STAN_MATH_REV_CORE_OPERATOR_UNARY_PLUS_HPP

#include <stan/math/rev/core/var.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/rev/core/precomp_v_vari.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {
  namespace math {

    /**
     * Unary plus operator for variables (C++).
     *
     * The function simply returns its input, because
     *
     * \f$\frac{d}{dx} +x = \frac{d}{dx} x = 1\f$.
     *
     * The effect of unary plus on a built-in C++ scalar type is
     * integer promotion.  Because variables are all
     * double-precision floating point already, promotion is
     * not necessary.
     *
       \f[
       \mbox{operator+}(x) =
       \begin{cases}
         x & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{operator+}(x)}{\partial x} =
       \begin{cases}
         1 & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Argument variable.
     * @return The input reference.
     */
    inline var operator+(const var& a) {
      if (unlikely(boost::math::isnan(a.vi_->val_)))
        return var(new precomp_v_vari(stan::math::NOT_A_NUMBER,
                                      a.vi_,
                                      stan::math::NOT_A_NUMBER));
      return a;
    }

  }
}
#endif
