#ifndef STAN__MATH__REV__SCAL__FUN__RISING_FACTORIAL_HPP
#define STAN__MATH__REV__SCAL__FUN__RISING_FACTORIAL_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/rising_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {
  namespace agrad {

    namespace {

      class rising_factorial_vv_vari : public op_vv_vari {
      public:
        rising_factorial_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::rising_factorial(avi->val_, bvi->val_), avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * stan::math::rising_factorial(avi_->val_, bvi_->val_) * (boost::math::digamma(avi_->val_ + bvi_->val_) - boost::math::digamma(avi_->val_));
          bvi_->adj_ += adj_ * stan::math::rising_factorial(avi_->val_, bvi_->val_) * boost::math::digamma(bvi_->val_ + avi_->val_);
        }
      };

      class rising_factorial_vd_vari : public op_vd_vari {
      public:
        rising_factorial_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::rising_factorial(avi->val_, b), avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * stan::math::rising_factorial(avi_->val_, bd_) * (boost::math::digamma(avi_->val_ + bd_) - boost::math::digamma(avi_->val_));
        }
      };

      class rising_factorial_dv_vari : public op_dv_vari {
      public:
        rising_factorial_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::rising_factorial(a, bvi->val_), a, bvi) {
        }
        void chain() {
          bvi_->adj_ += adj_ * stan::math::rising_factorial(ad_, bvi_->val_) * boost::math::digamma(bvi_->val_ + ad_);
        }
      };
    }

    inline var rising_factorial(const var& a,
                                const double& b) {
      return var(new rising_factorial_vd_vari(a.vi_, b));
    }

    inline var rising_factorial(const var& a,
                                const var& b) {
      return var(new rising_factorial_vv_vari(a.vi_, b.vi_));
    }

    inline var rising_factorial(const double& a,
                                const var& b) {
      return var(new rising_factorial_dv_vari(a, b.vi_));
    }
  }
}
#endif
