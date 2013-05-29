#ifndef __STAN__AGRAD__REV__RISING_FACTORIAL_HPP__
#define __STAN__AGRAD__REV__RISING_FACTORIAL_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/vd_vari.hpp>
#include <stan/math/functions/rising_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {
  namespace agrad {

    namespace {

      class rising_factorial_vd_vari : public op_vd_vari {
      public:
        rising_factorial_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::rising_factorial(avi->val_, b), avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * stan::math::rising_factorial(avi_->val_, bd_) * (boost::math::digamma(avi_->val_ + bd_) - boost::math::digamma(avi_->val_));
        }
      };
    }

    inline var rising_factorial(const var& a, 
                                 const double& b) {
      return var(new rising_factorial_vd_vari(a.vi_, b));
    }

  }
}
#endif
