#ifndef __STAN__DIFF__REV__INV_SQRT_HPP__
#define __STAN__DIFF__REV__INV_SQRT_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/functions/inv_sqrt.hpp>

namespace stan {
  namespace diff {
    
    namespace {
      class inv_sqrt_vari : public op_v_vari {
      public:
        inv_sqrt_vari(vari* avi) :
        op_v_vari(stan::math::inv_sqrt(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= 0.5 * adj_ / (avi_->val_ * std::sqrt(avi_->val_));
        }
      };
    }
    
    inline var inv_sqrt(const var& a) {
      return var(new inv_sqrt_vari(a.vi_));
    }

  }
}
#endif
