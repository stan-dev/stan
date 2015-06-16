#ifndef STAN_MATH_REV_CORE_OPERATOR_UNARY_NEGATIVE_HPP
#define STAN_MATH_REV_CORE_OPERATOR_UNARY_NEGATIVE_HPP

#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/v_vari.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <limits>

namespace stan {
  namespace math {

    namespace {
      class neg_vari : public op_v_vari {
      public:
        explicit neg_vari(vari* avi) :
          op_v_vari(-(avi->val_), avi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            avi_->adj_ -= adj_;
        }
      };
    }

    /**
     * Unary negation operator for variables (C++).
     *
     * \f$\frac{d}{dx} -x = -1\f$.
     *
       \f[
       \mbox{operator-}(x) =
       \begin{cases}
         -x & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{operator-}(x)}{\partial x} =
       \begin{cases}
         -1 & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Argument variable.
     * @return Negation of variable.
     */
    inline var operator-(const var& a) {
      return var(new neg_vari(a.vi_));
    }

  }
}
#endif
