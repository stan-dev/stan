#ifndef STAN__AGRAD__REV__FUNCTIONS__LGAMMA_HPP
#define STAN__AGRAD__REV__FUNCTIONS__LGAMMA_HPP

#include <valarray>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/constants.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace agrad {

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
    inline var lgamma(const stan::agrad::var& a) {
      double lgamma_a = boost::math::lgamma(a.val());
      return var(new lgamma_vari(lgamma_a, a.vi_));
    }

  }
}
#endif
