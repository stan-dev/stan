#ifndef __STAN__AGRAD__REV__INV_SQUARE_HPP__
#define __STAN__AGRAD__REV__INV_SQUARE_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <stan/math/functions/inv_square.hpp>

namespace stan {
  namespace agrad {
    
    namespace {
      class inv_square_vari : public op_v_vari {
      public:
        inv_square_vari(vari* avi) :
        op_v_vari(stan::math::inv_square(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= 2 * adj_ / (avi_->val_ * avi_->val_ * avi_->val_);
        }
      };
    }
    
    inline var inv_square(const var& a) {
      return var(new inv_square_vari(a.vi_));
    }

  }
}
#endif
