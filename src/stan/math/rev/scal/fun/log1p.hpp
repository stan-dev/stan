#ifndef STAN_MATH_REV_SCAL_FUN_LOG1P_HPP
#define STAN_MATH_REV_SCAL_FUN_LOG1P_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/log1p.hpp>
#include <valarray>

namespace stan {
  namespace math {

    namespace {
      class log1p_vari : public op_v_vari {
      public:
        explicit log1p_vari(vari* avi) :
          op_v_vari(stan::math::log1p(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (1 + avi_->val_);
        }
      };
    }

    /**
     * The log (1 + x) function for variables (C99).
     *
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \log (1 + x) = \frac{1}{1 + x}\f$.
     *
     * @param a The variable.
     * @return The log of 1 plus the variable.
     */
    inline var log1p(const stan::math::var& a) {
      return var(new log1p_vari(a.vi_));
    }

  }
}
#endif
