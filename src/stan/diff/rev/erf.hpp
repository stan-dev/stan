#ifndef __STAN__DIFF__REV__ERF_HPP__
#define __STAN__DIFF__REV__ERF_HPP__

#include <valarray>
#include <boost/math/special_functions/erf.hpp>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/constants.hpp>

namespace stan {
  namespace diff {

    namespace {
      class erf_vari : public op_v_vari {
      public:
        erf_vari(vari* avi) :
          op_v_vari(boost::math::erf(avi->val_),avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * stan::math::TWO_OVER_SQRT_PI * std::exp(- avi_->val_ * avi_->val_);
        }
      };
    }

    /**
     * The error function for variables (C99).
     *
     * For non-variable function, see boost::math::erf()
     *
     * The derivative is
     *
     * \f$\frac{d}{dx} \mbox{erf}(x) = \frac{2}{\sqrt{\pi}} \exp(-x^2)\f$.
     * 
     * @param a The variable.
     * @return Error function applied to the variable.
     */
    inline var erf(const stan::diff::var& a) {
      return var(new erf_vari(a.vi_));
    }

  }
}
#endif
