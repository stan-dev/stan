#ifndef __STAN__AGRAD__REV__LOG1M_HPP__
#define __STAN__AGRAD__REV__LOG1M_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/op/v_vari.hpp>
#include <stan/math/functions/log1p.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class log1m_vari : public op_v_vari {
      public:
        log1m_vari(vari* avi) :
          op_v_vari(stan::math::log1p(-avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ / (avi_->val_ - 1);
        }
      };
    }

    /**
     * The log (1 - x) function for variables.
     *
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \log (1 - x) = -\frac{1}{1 - x}\f$.
     *
     * @param a The variable.
     * @return The variable representing log of 1 minus the variable.
     */
    inline var log1m(const stan::agrad::var& a) {
      return var(new log1m_vari(a.vi_));
    }

  }
}
#endif
