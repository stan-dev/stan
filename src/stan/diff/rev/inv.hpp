#ifndef __STAN__DIFF__REV__INV_HPP__
#define __STAN__DIFF__REV__INV_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/functions/inv.hpp>

namespace stan {
  namespace diff {
    
    namespace {
      class inv_vari : public op_v_vari {
      public:
        inv_vari(vari* avi) :
        op_v_vari(stan::math::inv(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ -= adj_ / (avi_->val_ * avi_->val_);
        }
      };
    }
    
    inline var inv(const var& a) {
      return var(new inv_vari(a.vi_));
    }

  }
}
#endif
