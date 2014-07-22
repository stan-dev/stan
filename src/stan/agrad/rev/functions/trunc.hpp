#ifndef STAN__AGRAD__REV__FUNCTIONS__TRUNC_HPP
#define STAN__AGRAD__REV__FUNCTIONS__TRUNC_HPP

#include <boost/math/special_functions/trunc.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>

namespace stan {
  namespace agrad {

    namespace {
      // derivative 0 almost everywhere
      class trunc_vari : public vari {
      public:
        trunc_vari(vari* avi) :
          vari(boost::math::trunc(avi->val_)) { 
        }
      };
    }

    /**
     * Returns the truncatation of the specified variable (C99).
     *
     * See boost::math::trunc() for the double-based version.
     *
     * The derivative is zero everywhere but at integer values, so for
     * convenience the derivative is defined to be everywhere zero,
     *
     * \f$\frac{d}{dx} \mbox{trunc}(x) = 0\f$.
     *
     * @param a Specified variable.
     * @return Truncation of the variable.
     */
    inline var trunc(const stan::agrad::var& a) {
      return var(new trunc_vari(a.vi_));
    }

  }
}
#endif
