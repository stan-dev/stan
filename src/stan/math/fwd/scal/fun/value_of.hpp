#ifndef STAN__MATH__FWD__SCAL__FUN__VALUE_OF_HPP
#define STAN__MATH__FWD__SCAL__FUN__VALUE_OF_HPP

#include <stan/math/fwd/core.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the value of the specified variable.  
     *
     * @param v Variable.
     * @return Value of variable.
     */
    template<typename T>
    inline T value_of(const fvar<T>& v) {
      return v.val_;
    }
    
  }
}
#endif
