#ifndef STAN__AGRAD__REV__FUNCTIONS__IS_UNINITIALIZED_HPP
#define STAN__AGRAD__REV__FUNCTIONS__IS_UNINITIALIZED_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/operators/operator_unary_negative.hpp>
#include <stan/math/functions/is_uninitialized.hpp>

namespace stan {

  namespace agrad {

    /**
     * Returns <code>true</code> if the specified variable is
     * uninitialized.
     * 
     * This overload of the
     * <code>stan::math::is_uninitialized()</code> function delegates
     * the return to the <code>is_uninitialized()</code> method on the
     * specified variable.
     *
     * @param x Object to test.
     * @return <code>true</code> if the specified object is uninitialized.
     */
    inline bool is_uninitialized(var x) {
      return x.is_uninitialized();
    }

  }
}
#endif
