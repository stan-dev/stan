#ifndef STAN_MATH_REV_CORE_EMPTY_NESTED_HPP
#define STAN_MATH_REV_CORE_EMPTY_NESTED_HPP

#include <stan/math/rev/core/chainable.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>

namespace stan {
  namespace math {

    /**
     * Return true if there is no nested autodiff being executed.
     */
    static inline bool empty_nested() {
      return ChainableStack::nested_var_stack_sizes_.empty();
    }


  }
}
#endif
