#ifndef STAN__AGRAD__REV__FUNCTIONS__LOG_RISING_FACTORIAL_HPP
#define STAN__AGRAD__REV__FUNCTIONS__LOG_RISING_FACTORIAL_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/vd_vari.hpp>
#include <stan/agrad/rev/internal/dv_vari.hpp>
#include <stan/agrad/rev/internal/vv_vari.hpp>
#include <stan/math/functions/log_rising_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {
  namespace agrad {

    namespace {

      class log_rising_factorial_vv_vari : public op_vv_vari {
      public:
        log_rising_factorial_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::log_rising_factorial(avi->val_, bvi->val_), avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * (boost::math::digamma(avi_->val_ + bvi_->val_) 
                                - boost::math::digamma(avi_->val_));
          bvi_->adj_ += adj_ * boost::math::digamma(avi_->val_ + bvi_->val_);
        }
      };

      class log_rising_factorial_vd_vari : public op_vd_vari {
      public:
        log_rising_factorial_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::log_rising_factorial(avi->val_, b), avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * (boost::math::digamma(avi_->val_ + bd_) - boost::math::digamma(avi_->val_));
        }
      };

      class log_rising_factorial_dv_vari : public op_dv_vari {
      public:
        log_rising_factorial_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::log_rising_factorial(a, bvi->val_), a, bvi) {
        }
        void chain() {
          bvi_->adj_ += adj_ * boost::math::digamma(bvi_->val_ + ad_);
        }
      };
    }

    inline var log_rising_factorial(const var& a, 
                                    const double& b) {
      return var(new log_rising_factorial_vd_vari(a.vi_, b));
    }

    inline var log_rising_factorial(const var& a, 
                                    const var& b) {
      return var(new log_rising_factorial_vv_vari(a.vi_, b.vi_));
    }

    inline var log_rising_factorial(const double& a, 
                                    const var& b) {
      return var(new log_rising_factorial_dv_vari(a, b.vi_));
    }
  }
}
#endif
