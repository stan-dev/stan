#ifndef STAN__MATH__REV__CORE__EMPTY_NESTED_HPP
#define STAN__MATH__REV__CORE__EMPTY_NESTED_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return true if there is no nested autodiff being executed.
     */
    static inline bool empty_nested() {
      return ChainableStack::nested_var_stack_sizes_.empty();
    }


  }
}
#endif
