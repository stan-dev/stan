#ifndef STAN_MATH_REV_SCAL_FUN_LOG1M_EXP_HPP
#define STAN_MATH_REV_SCAL_FUN_LOG1M_EXP_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/log1m_exp.hpp>
#include <stan/math/rev/scal/fun/calculate_chain.hpp>
#include <cmath>

namespace stan {
  namespace math {

    namespace {
      class log1m_exp_v_vari : public op_v_vari {
      public:
        explicit log1m_exp_v_vari(vari* avi) :
          op_v_vari(stan::math::log1m_exp(avi->val_),
                    avi) {
        }
        void chain() {
          // derivative of
          //   log(1-exp(x)) = -exp(x)/(1-exp(x))
          //                 = -1/(exp(-x)-1)
          //                 = -1/expm1(-x)
          avi_->adj_ -= adj_ / ::expm1(-(avi_->val_));
        }
      };
    }

    /**
     * Return the log of 1 minus the exponential of the specified
     * variable.
     */
    inline var log1m_exp(const stan::math::var& a) {
      return var(new log1m_exp_v_vari(a.vi_));
    }

  }
}
#endif
