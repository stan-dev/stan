#ifndef __STAN__DIFF__REV__TGAMMA_HPP__
#define __STAN__DIFF__REV__TGAMMA_HPP__

#include <boost/math/special_functions/digamma.hpp>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {

    namespace {
      class tgamma_vari : public op_v_vari {
      public:
        tgamma_vari(vari* avi) :
          op_v_vari(boost::math::tgamma(avi->val_), avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * val_ * boost::math::digamma(avi_->val_);
        }
      };
    }

    /**
     * Return the Gamma function applied to the specified variable (C99).
     *
     * See boost::math::tgamma() for the double-based version.
     *
     * The derivative with respect to the argument is
     *
     * \f$\frac{d}{dx} \Gamma(x) = \Gamma(x) \Psi^{(0)}(x)\f$
     *
     * where \f$\Psi^{(0)}(x)\f$ is the digamma function.
     *
     * See boost::math::digamma() for the double-based version.
     *
     * @param a Argument to function.
     * @return The Gamma function applied to the specified argument.
     */
    inline var tgamma(const stan::diff::var& a) {
      return var(new tgamma_vari(a.vi_));
    }

  }
}
#endif
