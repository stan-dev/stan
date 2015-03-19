#ifndef STAN__MATH__REV__SCAL__FUN__LOG1P_EXP_HPP
#define STAN__MATH__REV__SCAL__FUN__LOG1P_EXP_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/log1p_exp.hpp>
#include <stan/math/rev/scal/fun/calculate_chain.hpp>

namespace stan {
  namespace agrad {

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
    inline var log1p_exp(const stan::agrad::var& a) {
      return var(new log1p_exp_v_vari(a.vi_));
    }

  }
}
#endif
