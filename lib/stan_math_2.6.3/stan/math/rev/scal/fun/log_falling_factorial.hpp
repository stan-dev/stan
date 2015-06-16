#ifndef STAN_MATH_REV_SCAL_FUN_LOG_FALLING_FACTORIAL_HPP
#define STAN_MATH_REV_SCAL_FUN_LOG_FALLING_FACTORIAL_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/log_falling_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <limits>

namespace stan {
  namespace math {

    namespace {

      class log_falling_factorial_vv_vari : public op_vv_vari {
      public:
        log_falling_factorial_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::log_falling_factorial(avi->val_, bvi->val_),
                     avi, bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bvi_->val_))) {
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          } else {
            avi_->adj_ += adj_ * boost::math::digamma(avi_->val_ + 1);
            bvi_->adj_ += adj_ * -boost::math::digamma(bvi_->val_ + 1);
          }
        }
      };

      class log_falling_factorial_vd_vari : public op_vd_vari {
      public:
        log_falling_factorial_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::log_falling_factorial(avi->val_, b), avi, b) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(avi_->val_)
                       || boost::math::isnan(bd_)))
            avi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            avi_->adj_ += adj_ * boost::math::digamma(avi_->val_ + 1);
        }
      };

      class log_falling_factorial_dv_vari : public op_dv_vari {
      public:
        log_falling_factorial_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::log_falling_factorial(a, bvi->val_), a, bvi) {
        }
        void chain() {
          if (unlikely(boost::math::isnan(ad_)
                       || boost::math::isnan(bvi_->val_)))
            bvi_->adj_ = std::numeric_limits<double>::quiet_NaN();
          else
            bvi_->adj_ += adj_ * -boost::math::digamma(bvi_->val_ + 1);
        }
      };
    }

    inline var log_falling_factorial(const var& a,
                                     const double& b) {
      return var(new log_falling_factorial_vd_vari(a.vi_, b));
    }

    inline var log_falling_factorial(const var& a,
                                     const var& b) {
      return var(new log_falling_factorial_vv_vari(a.vi_, b.vi_));
    }

    inline var log_falling_factorial(const double& a,
                                     const var& b) {
      return var(new log_falling_factorial_dv_vari(a, b.vi_));
    }
  }
}
#endif
