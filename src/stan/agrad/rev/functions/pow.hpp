#ifndef STAN__AGRAD__REV__FUNCTIONS__POW_HPP
#define STAN__AGRAD__REV__FUNCTIONS__POW_HPP

#include <cmath>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <stan/agrad/rev/functions/sqrt.hpp>
#include <stan/agrad/rev/operators/operator_multiplication.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class pow_vv_vari : public op_vv_vari {
      public:
        pow_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(std::pow(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bvi_->val_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            if (avi_->val_ == 0.0) return; // partials zero, avoids 0 & log(0)
            avi_->adj_ += adj_ * bvi_->val_ * val_ / avi_->val_;
            bvi_->adj_ += adj_ * std::log(avi_->val_) * val_;
          }
        }
      };

      class pow_vd_vari : public op_vd_vari {
      public:
        pow_vd_vari(vari* avi, double b) :
          op_vd_vari(std::pow(avi->val_,b),avi,b) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bd_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else {
            if (avi_->val_ == 0.0) return; // partials zero, avoids 0 & log(0)
            avi_->adj_ += adj_ * bd_ * val_ / avi_->val_;
          }
        }
      };

      class pow_dv_vari : public op_dv_vari {
      public:
        pow_dv_vari(double a, vari* bvi) :
          op_dv_vari(std::pow(a,bvi->val_),a,bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(bvi_->val_)
                       || boost::math::isnan(ad_)))
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else {
            if (ad_ == 0.0) return; // partials zero, avoids 0 & log(0)
            bvi_->adj_ += adj_ * std::log(ad_) * val_;
          }
        }
      };
    }

    /**
     * Return the base raised to the power of the exponent (cmath).
     *
     * The partial derivatives are
     *
     * \f$\frac{\partial}{\partial x} \mbox{pow}(x,y) = y x^{y-1}\f$, and
     *
     * \f$\frac{\partial}{\partial y} \mbox{pow}(x,y) = x^y \ \log x\f$.
     *
     *
       \f[
       \mbox{pow}(x,y) = 
       \begin{cases}
         x^y & \mbox{if } -\infty\leq x,y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{pow}(x,y)}{\partial x} = 
       \begin{cases}
         yx^{y-1} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{pow}(x,y)}{\partial y} = 
       \begin{cases}
         x^y\ln x & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param base Base variable.
     * @param exponent Exponent variable.
     * @return Base raised to the exponent.
     */
    inline var pow(const var& base, const var& exponent) {
      return var(new pow_vv_vari(base.vi_,exponent.vi_));
    }
  
    /**
     * Return the base variable raised to the power of the exponent
     * scalar (cmath).
     *
     * The derivative for the variable is
     *
     * \f$\frac{d}{dx} \mbox{pow}(x,c) = c x^{c-1}\f$.
     *
     * @param base Base variable.
     * @param exponent Exponent scalar.
     * @return Base raised to the exponent.
     */
    inline var pow(const var& base, const double exponent) {
      if (exponent == 0.5)
        return sqrt(base);
      if (exponent == 1.0)
        return base;
      if (exponent == 2.0)
        return base * base; // FIXME: square() functionality from special_functions
      return var(new pow_vd_vari(base.vi_,exponent));
    }

    /**
     * Return the base scalar raised to the power of the exponent
     * variable (cmath).
     *
     * The derivative for the variable is
     * 
     * \f$\frac{d}{d y} \mbox{pow}(c,y) = c^y \log c \f$.
     * 
     * @param base Base scalar.
     * @param exponent Exponent variable.
     * @return Base raised to the exponent.
     */
    inline var pow(const double base, const var& exponent) {
      return var(new pow_dv_vari(base,exponent.vi_));
    }

  }
}
#endif
