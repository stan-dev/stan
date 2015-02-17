#ifndef STAN__MATH__FWD__SCAL__FUN__VALUE_OF_REC_HPP
#define STAN__MATH__FWD__SCAL__FUN__VALUE_OF_REC_HPP

#include <stan/math/prim/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/core/fvar.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the value of the specified variable.  
     *
     * T must implement value_of_rec.
     *
     * @tparam T Scalar type
     * @param v Variable.
     * @return Value of variable.
     */

    template <typename T>
    inline double value_of_rec(const fvar<T>& v) {
      using stan::math::value_of_rec;
      return value_of_rec(v.val_);
    }

    
  }
}
#endif
