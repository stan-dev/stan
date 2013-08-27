#ifndef __STAN__DIFF__REV__LGAMMA_HPP__
#define __STAN__DIFF__REV__LGAMMA_HPP__

#include <valarray>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>
#include <stan/math/constants.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace diff {

    namespace {
      class lgamma_vari : public op_v_vari {
      public:
        lgamma_vari(double value, vari* avi) :
          op_v_vari(value, avi) {
        }
        void chain() {
          avi_->adj_ += adj_ * boost::math::digamma(avi_->val_);
        }
      };
    }

    /**
     * The log gamma function for variables (C99).  
     *
     * The derivatie is the digamma function,
     *
     * \f$\frac{d}{dx} \Gamma(x) = \psi^{(0)}(x)\f$.
     * 
     * @param a The variable.
     * @return Log gamma of the variable.
     */
    inline var lgamma(const stan::diff::var& a) {
      double lgamma_a = boost::math::lgamma(a.val());
      return var(new lgamma_vari(lgamma_a, a.vi_));
    }

  }
}
#endif
