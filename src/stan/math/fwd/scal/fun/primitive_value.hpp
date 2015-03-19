#ifndef STAN__MATH__FWD__SCAL__FUN__PRIMITIVE_VALUE_HPP
#define STAN__MATH__FWD__SCAL__FUN__PRIMITIVE_VALUE_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/primitive_value.hpp>

namespace stan {

  namespace agrad {


    /**
     * Return the primitive value of the specified forward-mode
     * autodiff variable.  This function applies recursively to
     * higher-order autodiff types to return a primitive double value.
     *
     * @tparam T scalar type for autodiff variable.
     * @param v input variable.
     * @return primitive value of input.
     */
    template <typename T>
    inline double primitive_value(const fvar<T>& v) {
      using stan::math::primitive_value;
      return primitive_value(v.val_);
    }


  }

}

#endif
