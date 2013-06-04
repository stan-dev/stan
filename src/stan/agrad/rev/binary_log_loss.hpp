#ifndef __STAN__AGRAD__REV__BINARY_LOG_LOSS_HPP__
#define __STAN__AGRAD__REV__BINARY_LOG_LOSS_HPP__

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/dv_vari.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/binary_log_loss.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class binary_log_loss_dv_vari : public op_dv_vari {
      public:
        binary_log_loss_dv_vari(int a, vari* bvi) :
          op_dv_vari(stan::math::binary_log_loss(a, bvi->val_),a,bvi) {
        }
        void chain() {
          if (ad_ == 0)
            bvi_->adj_ += adj_ / bvi_->val_;
          if (ad_ == 1)
            bvi_->adj_ -= adj_ / bvi_->val_;
        }
      };
    }

    inline var binary_log_loss(int a, const stan::agrad::var& b) {
      return var(new binary_log_loss_dv_vari(a, b.vi_));
    }

  }
}
#endif
