#ifndef __STAN__DIFF__REV__FLOOR_HPP__
#define __STAN__DIFF__REV__FLOOR_HPP__

#include <cmath>
#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/op/v_vari.hpp>

namespace stan {
  namespace diff {

    namespace {
      class floor_vari : public vari {
      public:
        floor_vari(vari* avi) :
          vari(std::floor(avi->val_)) {
        }
      };
    }
    
    /**
     * Return the floor of the specified variable (cmath).  
     *
     * The derivative of the floor function is defined and
     * zero everywhere but at integers, so we set these derivatives
     * to zero for convenience, 
     *
     * \f$\frac{d}{dx} {\lfloor x \rfloor} = 0\f$.
     *
     * The floor function rounds down.  For double values, this is the largest
     * integral value that is not greater than the specified value.
     * Although this function is not differentiable because it is
     * discontinuous at integral values, its gradient is returned as
     * zero everywhere.
     * 
     * @param a Input variable.
     * @return Floor of the variable.
     */
    inline var floor(const var& a) {
      return var(new floor_vari(a.vi_));
    }

  }
}
#endif
