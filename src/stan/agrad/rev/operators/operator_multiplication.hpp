#ifndef STAN__AGRAD__REV__OPERATORS__OPERATOR_MULTIPLICATION_HPP
#define STAN__AGRAD__REV__OPERATORS__OPERATOR_MULTIPLICATION_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class multiply_vv_vari : public op_vv_vari {
      public:
        multiply_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ * bvi->val_, avi, bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bvi_->val_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            avi_->adj_ += bvi_->val_ * adj_;
            bvi_->adj_ += avi_->val_ * adj_;
          }
        }
      };

      class multiply_vd_vari : public op_vd_vari {
      public:
        multiply_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ * b, avi, b) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bd_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            avi_->adj_ += adj_ * bd_;
        }
      };
    }

    /**
     * Multiplication operator for two variables (C++).
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} (x * y) = y\f$, and
     *
     * \f$\frac{\partial}{\partial y} (x * y) = x\f$.
     *
       \f[
       \mbox{operator*}(x,y) = 
       \begin{cases}
         xy & \mbox{if } -\infty\leq x,y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{operator*}(x,y)}{\partial x} = 
       \begin{cases}
         y & \mbox{if } -\infty\leq x,y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{operator*}(x,y)}{\partial y} = 
       \begin{cases}
         x & \mbox{if } -\infty\leq x,y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a First variable operand.
     * @param b Second variable operand.
     * @return Variable result of multiplying operands.
     */
    inline var operator*(const var& a, const var& b) {
      return var(new multiply_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Multiplication operator for a variable and a scalar (C++).
     *
     * The partial derivative for the variable is
     *
     * \f$\frac{\partial}{\partial x} (x * c) = c\f$, and
     * 
     * @param a Variable operand.
     * @param b Scalar operand.
     * @return Variable result of multiplying operands.
     */
    inline var operator*(const var& a, const double b) {
      if (b == 1.0)
        return a;
      return var(new multiply_vd_vari(a.vi_,b));
    }

    /**
     * Multiplication operator for a scalar and a variable (C++).
     *
     * The partial derivative for the variable is
     *
     * \f$\frac{\partial}{\partial y} (c * y) = c\f$.
     *
     * @param a Scalar operand.
     * @param b Variable operand.
     * @return Variable result of multiplying the operands.
     */
    inline var operator*(const double a, const var& b) {
      if (a == 1.0)
        return b;
      return var(new multiply_vd_vari(b.vi_,a)); // by symmetry
    }

  }
}
#endif
