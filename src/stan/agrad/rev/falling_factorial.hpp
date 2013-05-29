#ifndef __STAN__AGRAD__REV__FALLING_FACTORIAL_HPP__
#define __STAN__AGRAD__REV__FALLING_FACTORIAL_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/vd_vari.hpp>
#include <stan/math/functions/falling_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {
  namespace agrad {

    namespace {

      class falling_factorial_vd_vari : public op_vd_vari {
      public:
        falling_factorial_vd_vari(vari* avi, double b) :
          op_vd_vari(stan::math::falling_factorial(avi->val_, b), avi, b) {
        }
        void chain() {
          avi_->adj_ += adj_ * stan::math::falling_factorial(avi_->val_, bd_) * boost::math::digamma(avi_->val_ + 1);
        }
      };
    }

    inline var falling_factorial(const var& a, 
                                 const double& b) {
      return var(new falling_factorial_vd_vari(a.vi_, b));
    }

  }
}
#endif
