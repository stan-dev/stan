#ifndef STAN_MATH_REV_CORE_NESTED_SIZE_HPP
#define STAN_MATH_REV_CORE_NESTED_SIZE_HPP

#include <stan/math/rev/core/chainable.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <cstdlib>

namespace stan {
  namespace math {

    static inline size_t nested_size() {
      return ChainableStack::var_stack_.size()
        - ChainableStack::nested_var_stack_sizes_.back();
    }

  }
}
#endif
