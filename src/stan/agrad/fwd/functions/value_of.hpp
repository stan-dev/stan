#ifndef STAN__AGRAD__FWD__FUNCTIONS__VALUE_OF_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__VALUE_OF_HPP

#include <stan/agrad/fwd/fvar.hpp>

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
