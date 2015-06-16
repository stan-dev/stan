#ifndef STAN_MATH_REV_SCAL_FUN_FDIM_HPP
#define STAN_MATH_REV_SCAL_FUN_FDIM_HPP

#include <stan/math/rev/core.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/prim/scal/meta/likely.hpp>
#include <limits>

namespace stan {
  namespace math {

    namespace {
      class fdim_vv_vari : public op_vv_vari {
      public:
        fdim_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(avi->val_ - bvi->val_, avi, bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bvi_->val_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            avi_->adj_ += adj_;
            bvi_->adj_ -= adj_;
          }
        }
      };

      class fdim_vd_vari : public op_vd_vari {
      public:
        fdim_vd_vari(vari* avi, double b) :
          op_vd_vari(avi->val_ - b, avi, b) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bd_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            avi_->adj_ += adj_;
        }
      };

      class fdim_dv_vari : public op_dv_vari {
      public:
        fdim_dv_vari(double a, vari* bvi) :
          op_dv_vari(a - bvi->val_, a, bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(bvi_->val_)
                       || boost::math::isnan(ad_)))
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            bvi_->adj_ -= adj_;
        }
      };
    }

    /**
     * Return the positive difference between the first variable's the value
     * and the second's (C99).
     *
     * See stan::math::fdim() for the double-based version.
     *
     * The partial derivative with respect to the first argument is
     *
     * \f$\frac{\partial}{\partial x} \mbox{fdim}(x, y) = 0.0\f$ if \f$x < y\f$, and
     *
     * \f$\frac{\partial}{\partial x} \mbox{fdim}(x, y) = 1.0\f$ if \f$x \geq y\f$.
     *
     * With respect to the second argument, the partial is
     *
     * \f$\frac{\partial}{\partial y} \mbox{fdim}(x, y) = 0.0\f$ if \f$x < y\f$, and
     *
     * \f$\frac{\partial}{\partial y} \mbox{fdim}(x, y) = -\lfloor\frac{x}{y}\rfloor\f$ if \f$x \geq y\f$.
     *
     *
       \f[
       \mbox{fdim}(x, y) =
       \begin{cases}
         0 & \mbox{if } x < y\\
         x-y & \mbox{if } x \geq y \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{fdim}(x, y)}{\partial x} =
       \begin{cases}
         0 & \mbox{if } x < y \\
         1 & \mbox{if } x \geq y \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{fdim}(x, y)}{\partial y} =
       \begin{cases}
         0 & \mbox{if } x < y \\
         -\lfloor\frac{x}{y}\rfloor & \mbox{if } x \geq y \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a First variable.
     * @param b Second variable.
     * @return The positive difference between the first and second
     * variable.
     */
    inline var fdim(const stan::math::var& a,
                    const stan::math::var& b) {
      if (!(a.vi_->val_ <= b.vi_->val_))
        return var(new fdim_vv_vari(a.vi_, b.vi_));
      else
        return var(new vari(0.0));
    }

    /**
     * Return the positive difference between the first value and the
     * value of the second variable (C99).
     *
     * See fdim(var, var) for definitions of values and derivatives.
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d y} \mbox{fdim}(c, y) = 0.0\f$ if \f$c < y\f$, and
     *
     * \f$\frac{d}{d y} \mbox{fdim}(c, y) = -\lfloor\frac{c}{y}\rfloor\f$ if \f$c \geq y\f$.
     *
     * @param a First value.
     * @param b Second variable.
     * @return The positive difference between the first and second
     * arguments.
     */
    inline var fdim(const double& a,
                    const stan::math::var& b) {
      return a <= b.vi_->val_
        ? var(new vari(0.0))
        : var(new fdim_dv_vari(a, b.vi_));
    }

    /**
     * Return the positive difference between the first variable's value
     * and the second value (C99).
     *
     * See fdim(var, var) for definitions of values and derivatives.
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d x} \mbox{fdim}(x, c) = 0.0\f$ if \f$x < c\f$, and
     *
     * \f$\frac{d}{d x} \mbox{fdim}(x, c) = 1.0\f$ if \f$x \geq yc\f$.
     *
     * @param a First value.
     * @param b Second variable.
     * @return The positive difference between the first and second arguments.
     */
    inline var fdim(const stan::math::var& a,
                    const double& b) {
      return a.vi_->val_ <= b
        ? var(new vari(0.0))
        : var(new fdim_vd_vari(a.vi_, b));
    }

  }
}
#endif
