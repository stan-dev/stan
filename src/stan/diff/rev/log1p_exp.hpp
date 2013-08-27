#ifndef __STAN__DIFF__REV__LOG1P_EXP_HPP__
#define __STAN__DIFF__REV__LOG1P_EXP_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/functions/log1p_exp.hpp>
#include <stan/diff/rev/calculate_chain.hpp>

namespace stan {
  namespace diff {

    namespace {
      class log1p_exp_v_vari : public op_v_vari {
      public:
        log1p_exp_v_vari(vari* avi) :
          op_v_vari(stan::math::log1p_exp(avi->val_),
                    avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * calculate_chain(avi_->val_, val_);
        }
      };      
    }

    /**
     * Return the log of 1 plus the exponential of the specified
     * variable.
     */
    inline var log1p_exp(const stan::diff::var& a) {
      return var(new log1p_exp_v_vari(a.vi_));
    }

  }
}
#endif
