#ifndef STAN__AGRAD__FWD__FUNCTIONS__VALUE_OF_REC_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__VALUE_OF_REC_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/math/functions/value_of_rec.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the value of the specified variable.  
     *
     * @tparam T Scalar type
     * @param v Variable.
     * @return Value of variable.
     */

    template<typename T>
    inline double value_of_rec(const fvar<T>& v) {
      using stan::math::value_of_rec;
      return value_of_rec(v.val_);
    }

    
  }
}
#endif
