#ifndef __STAN__DIFF__REV__ROUND_HPP__
#define __STAN__DIFF__REV__ROUND_HPP__

#include <boost/math/special_functions/round.hpp>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/vari.hpp>

namespace stan {
  namespace diff {

    namespace {
      // derivative 0 almost everywhere
      class round_vari : public vari {
      public:
        round_vari(vari* avi) :
          vari(boost::math::round(avi->val_)) {
        }
      };
    }

    /**
     * Returns the rounded form of the specified variable (C99).
     *
     * See boost::math::round() for the double-based version.
     *
     * The derivative is zero everywhere but numbers half way between
     * whole numbers, so for convenience the derivative is defined to
     * be everywhere zero,
     *
     * \f$\frac{d}{dx} \mbox{round}(x) = 0\f$.
     *
     * @param a Specified variable.
     * @return Rounded variable.
     */
    inline var round(const stan::diff::var& a) {
      return var(new round_vari(a.vi_));
    }

  }
}
#endif
