#ifndef STAN__AGRAD__REV__FUNCTIONS__ACOSH_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ACOSH_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <boost/math/special_functions/acosh.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>
#include <stan/agrad/rev/operators/operator_greater_than.hpp>
#include <stan/agrad/rev/operators/operator_equal.hpp>
#include <stan/agrad/rev/operators/operator_unary_negative.hpp>

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
     * @param a The variable.
     * @return Inverse hyperbolic cosine of the variable.
     */
    inline var acosh(const stan::agrad::var& a) {
      if (boost::math::isinf(a) && a > 0.0)
        return var(new acosh_vari(a.val(),a.vi_));
      return var(new acosh_vari(boost::math::acosh(a.val()),a.vi_));
    }

  }
}

#endif
