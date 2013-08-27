#ifndef __STAN__DIFF__REV__CEIL_HPP__
#define __STAN__DIFF__REV__CEIL_HPP__

#include <cmath>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {

    namespace {
      class ceil_vari : public vari {
      public:
        ceil_vari(vari* avi) :
          vari(std::ceil(avi->val_)) {
        }
      };
    }
    
    /**
     * Return the ceiling of the specified variable (cmath).
     *
     * The derivative of the ceiling function is defined and
     * zero everywhere but at integers, and we set them to zero for
     * convenience, 
     *
     * \f$\frac{d}{dx} {\lceil x \rceil} = 0\f$.
     *
     * The ceiling function rounds up.  For double values, this is the
     * smallest integral value that is not less than the specified
     * value.  Although this function is not differentiable because it
     * is discontinuous at integral values, its gradient is returned
     * as zero everywhere.
     * 
     * @param a Input variable.
     * @return Ceiling of the variable.
     */
    inline var ceil(const var& a) {
      return var(new ceil_vari(a.vi_));
    }

  }
}
#endif
