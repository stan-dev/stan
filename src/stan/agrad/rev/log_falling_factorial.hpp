#ifndef __STAN__AGRAD__REV__LOG_FALLING_FACTORIAL_HPP__
#define __STAN__AGRAD__REV__LOG_FALLING_FACTORIAL_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/vd_vari.hpp>
#include <stan/agrad/rev/op/dv_vari.hpp>
#include <stan/agrad/rev/op/vv_vari.hpp>
#include <stan/math/functions/log_falling_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {
  namespace agrad {

    namespace {

     class log_falling_factorial_vv_vari : public op_vv_vari {
      public:
        log_falling_factorial_vv_vari(vari* avi, vari* bvi) :
          op_vv_vari(stan::math::log_falling_factorial(avi->val_, bvi->val_), avi, bvi) {
        }
        void chain() {
          avi_->adj_ += adj_ * boost::math::digamma(avi_->val_ + 1);
          bvi_->adj_ += adj_ * -boost::math::digamma(bvi_->val_ + 1);
        }
      };

      class log_falling_factorial_vd_vari : public op_vd_vari {
      public:
        log_falling_factorial_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::log_falling_factorial(avi->val_, b), avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * boost::math::digamma(avi_->val_ + 1);
        }
      };

      class log_falling_factorial_dv_vari : public op_dv_vari {
      public:
        log_falling_factorial_dv_vari(double a, vari* bvi) :
          op_dv_vari(stan::math::log_falling_factorial(a, bvi->val_), a, bvi) {
        }
        void chain() {
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
