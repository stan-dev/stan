#ifndef STAN__MATH__REV__SCAL__FUN__INV_CLOGLOG_HPP
#define STAN__MATH__REV__SCAL__FUN__INV_CLOGLOG_HPP

#include <stan/math/rev/arr/meta/var.hpp>
#include <stan/math/prim/scal/fun/inv_cloglog.hpp>
#include <stan/math/rev/scal/fun/v_vari.hpp>

namespace stan {
  namespace agrad {

    namespace {
      class inv_cloglog_vari : public op_v_vari {
      public:
        inv_cloglog_vari(vari* avi) :
          op_v_vari(stan::math::inv_cloglog(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * std::exp(avi_->val_ - std::exp(avi_->val_));
        }
      };
    }

    /**
     * Return the inverse complementary log-log function applied
     * specified variable (stan).
     *
     * See stan::math::inv_cloglog() for the double-based version.
     *
     * The derivative is given by
     *
     * \f$\frac{d}{dx} \mbox{cloglog}^{-1}(x) = \exp (x - \exp (x))\f$.
     *
     * @param a Variable argument.
     * @return The inverse complementary log-log of the specified
     * argument.
     */
    inline var inv_cloglog(const stan::agrad::var& a) {
      return var(new inv_cloglog_vari(a.vi_));
    }

  }
}
#endif
