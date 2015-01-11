#ifndef STAN__AGRAD__REV__OPERATORS__OPERATOR_DIVISION_HPP
#define STAN__AGRAD__REV__OPERATORS__OPERATOR_DIVISION_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace agrad {

    namespace {
      // (a/b)' = a' * (1 / b) - b' * (a / [b * b])
      class divide_vv_vari : public op_vv_vari {
      public:
        divide_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ / bvi->val_, avi, bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bvi_->val_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            avi_->adj_ += adj_ / bvi_->val_;
            bvi_->adj_ -= adj_ * avi_->val_ / (bvi_->val_ * bvi_->val_);
          }
        }
      };

      class divide_vd_vari : public op_vd_vari {
      public:
        divide_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ / b, avi, b) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bd_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            avi_->adj_ += adj_ / bd_;
        }
      };

      class divide_dv_vari : public op_dv_vari {
      public:
        divide_dv_vari(double a, vari* bvi) :
          op_dv_vari(a / bvi->val_, a, bvi) {
        }
        void chain() {
          bvi_->adj_ -= adj_ * ad_ / (bvi_->val_ * bvi_->val_);
        }
      };
    }

    /**
     * Division operator for two variables (C++).
     *
     * The partial derivatives for the variables are
     *
     * \f$\frac{\partial}{\partial x} (x/y) = 1/y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x/y) = -x / y^2\f$.
     *
       \f[
       \mbox{operator/}(x,y) = 
       \begin{cases}
         \frac{x}{y} & \mbox{if } -\infty\leq x,y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{operator/}(x,y)}{\partial x} = 
       \begin{cases}
         \frac{1}{y} & \mbox{if } -\infty\leq x,y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{operator/}(x,y)}{\partial y} = 
       \begin{cases}
         -\frac{x}{y^2} & \mbox{if } -\infty\leq x,y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of dividing the first variable by the
     * second.
     */
    inline var operator/(const var& a, const var& b) {
      return var(new divide_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Division operator for dividing a variable by a scalar (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{\partial}{\partial x} (x/c) = 1/c\f$.
     *
     * @param a Variable operand.
     * @param b Scalar operand.
     * @return Variable result of dividing the variable by the scalar.
     */
    inline var operator/(const var& a, const double b) {
      if (b == 1.0)
        return a;
      return var(new divide_vd_vari(a.vi_,b));
    }

    /**
     * Division operator for dividing a scalar by a variable (C++).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d y} (c/y) = -c / y^2\f$.
     * 
     * @param a Scalar operand.
     * @param b Variable operand.
     * @return Variable result of dividing the scalar by the variable.
     */
    inline var operator/(const double a, const var& b) {
      return var(new divide_dv_vari(a,b.vi_));
    }

  }
}
#endif
