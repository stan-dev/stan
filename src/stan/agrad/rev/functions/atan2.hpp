#ifndef STAN__AGRAD__REV__FUNCTIONS__ATAN2_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ATAN2_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <math.h>

namespace stan {
  namespace agrad {
    
    namespace {
      class atan2_vv_vari : public op_vv_vari {
      public:
        atan2_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(::atan2(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          double a_sq_plus_b_sq = (avi_->val_ * avi_->val_) + (bvi_->val_ * bvi_->val_);
          avi_->adj_ += adj_ * bvi_->val_ / a_sq_plus_b_sq;
          bvi_->adj_ -= adj_ * avi_->val_ / a_sq_plus_b_sq;
        }
      };

      class atan2_vd_vari : public op_vd_vari {
      public:
        atan2_vd_vari(vari* avi, double b) :
          op_vd_vari(::atan2(avi->val_,b),avi,b) {
        }
        void chain() {
          double a_sq_plus_b_sq = (avi_->val_ * avi_->val_) + (bd_ * bd_);
          avi_->adj_ += adj_ * bd_ / a_sq_plus_b_sq;
        }
      };

      class atan2_dv_vari : public op_dv_vari {
      public:
        atan2_dv_vari(double a, vari* bvi) :
          op_dv_vari(::atan2(a,bvi->val_),a,bvi) {
        }
        void chain() {
          double a_sq_plus_b_sq = (ad_ * ad_) + (bvi_->val_ * bvi_->val_);
          bvi_->adj_ -= adj_ * ad_ / a_sq_plus_b_sq;
        }
      };
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first variable divided by the second (cmath).
     *
     * The partial derivatives are defined by
     *
     * \f$ \frac{\partial}{\partial x} \arctan \frac{x}{y} = \frac{y}{x^2 + y^2}\f$, and
     * 
     * \f$ \frac{\partial}{\partial y} \arctan \frac{x}{y} = \frac{-x}{x^2 + y^2}\f$.
     *
     * @param a Numerator variable.
     * @param b Denominator variable.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const var& a, const var& b) {
      return var(new atan2_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first variable divided by the second scalar (cmath).
     *
     * The derivative with respect to the variable is
     *
     * \f$ \frac{d}{d x} \arctan \frac{x}{c} = \frac{c}{x^2 + c^2}\f$.
     *
     * @param a Numerator variable.
     * @param b Denominator scalar.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const var& a, const double b) {
      return var(new atan2_vd_vari(a.vi_,b));
    }

    /**
     * Return the principal value of the arc tangent, in radians, of
     * the first scalar divided by the second variable (cmath).
     *
     * The derivative with respect to the variable is
     *
     * \f$ \frac{\partial}{\partial y} \arctan \frac{c}{y} = \frac{-c}{c^2 + y^2}\f$.
     *
     *
       \f[
       \mbox{atan2}(x,y) = 
       \begin{cases}
         \arctan\left(\frac{x}{y}\right) & \mbox{if } -\infty\leq x \leq \infty, -\infty\leq y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{atan2}(x,y)}{\partial x} = 
       \begin{cases}
         \frac{y}{x^2+y^2} & \mbox{if } -\infty\leq x\leq \infty, -\infty\leq y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{atan2}(x,y)}{\partial y} = 
       \begin{cases}
         -\frac{x}{x^2+y^2} & \mbox{if } -\infty\leq x\leq \infty, -\infty\leq y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Numerator scalar.
     * @param b Denominator variable.
     * @return The arc tangent of the fraction, in radians.
     */
    inline var atan2(const double a, const var& b) {
      return var(new atan2_dv_vari(a,b.vi_));
    }

  }
}
#endif
