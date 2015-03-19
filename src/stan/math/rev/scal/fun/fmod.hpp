#ifndef STAN__MATH__REV__SCAL__FUN__FMOD_HPP
#define STAN__MATH__REV__SCAL__FUN__FMOD_HPP

#include <cmath>
#include <stan/math/rev/core.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class fmod_vv_vari : public op_vv_vari {
      public:
        fmod_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(std::fmod(avi->val_,bvi->val_),avi,bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bvi_->val_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            avi_->adj_ += adj_;
            bvi_->adj_ -= adj_ * static_cast<int>(avi_->val_ / bvi_->val_);
          }
        }
      };

      class fmod_vd_vari : public op_vd_vari {
      public:
        fmod_vd_vari(vari* avi, double b) :
          op_vd_vari(std::fmod(avi->val_,b),avi,b) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bd_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            avi_->adj_ += adj_;
        }
      };

      class fmod_dv_vari : public op_dv_vari {
      public:
        fmod_dv_vari(double a, vari* bvi) :
          op_dv_vari(std::fmod(a,bvi->val_),a,bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(bvi_->val_)
                       || boost::math::isnan(ad_)))
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else {
            int d = static_cast<int>(ad_ / bvi_->val_);
            bvi_->adj_ -= adj_ * d;
          }
        }
      };
    }

    /**
     * Return the floating point remainder after dividing the
     * first variable by the second (cmath).
     *
     * The partial derivatives with respect to the variables are defined
     * everywhere but where \f$x = y\f$, but we set these to match other values,
     * with
     *
     * \f$\frac{\partial}{\partial x} \mbox{fmod}(x,y) = 1\f$, and
     *
     * \f$\frac{\partial}{\partial y} \mbox{fmod}(x,y) = -\lfloor \frac{x}{y} \rfloor\f$.
     *
     *
       \f[
       \mbox{fmod}(x,y) =
       \begin{cases}
         x - \lfloor \frac{x}{y}\rfloor y & \mbox{if } -\infty\leq x,y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{fmod}(x,y)}{\partial x} =
       \begin{cases}
         1 & \mbox{if } -\infty\leq x,y\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{fmod}(x,y)}{\partial y} =
       \begin{cases}
         -\lfloor \frac{x}{y}\rfloor & \mbox{if } -\infty\leq x,y\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN or } y = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a First variable.
     * @param b Second variable.
     * @return Floating pointer remainder of dividing the first variable
     * by the second.
     */
    inline var fmod(const var& a, const var& b) {
      return var(new fmod_vv_vari(a.vi_,b.vi_));
    }

    /**
     * Return the floating point remainder after dividing the
     * the first variable by the second scalar (cmath).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d x} \mbox{fmod}(x,c) = \frac{1}{c}\f$.
     *
     * @param a First variable.
     * @param b Second scalar.
     * @return Floating pointer remainder of dividing the first variable by
     * the second scalar.
     */
    inline var fmod(const var& a, const double b) {
      return var(new fmod_vd_vari(a.vi_,b));
    }

    /**
     * Return the floating point remainder after dividing the
     * first scalar by the second variable (cmath).
     *
     * The derivative with respect to the variable is
     *
     * \f$\frac{d}{d y} \mbox{fmod}(c,y) = -\lfloor \frac{c}{y} \rfloor\f$.
     *
     * @param a First scalar.
     * @param b Second variable.
     * @return Floating pointer remainder of dividing first scalar by
     * the second variable.
     */
    inline var fmod(const double a, const var& b) {
      return var(new fmod_dv_vari(a,b.vi_));
    }

  }
}
#endif
