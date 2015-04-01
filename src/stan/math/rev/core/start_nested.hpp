#ifndef STAN__MATH__REV__CORE__START_NESTED_HPP
#define STAN__MATH__REV__CORE__START_NESTED_HPP

#include <stan/math/rev/core/chainablestack.hpp>

namespace stan {
  namespace agrad {

    /**
     * Record the current position so that <code>recover_memory_nested()</code>
     * can find it.
     */
    static inline void start_nested() {
      ChainableStack::nested_var_stack_sizes_
        .push_back(ChainableStack::var_stack_.size());
      ChainableStack::nested_var_nochain_stack_sizes_
        .push_back(ChainableStack::var_nochain_stack_.size());
      ChainableStack::nested_var_alloc_stack_starts_
        .push_back(ChainableStack::var_alloc_stack_.size());
      ChainableStack::memalloc_.start_nested();
    }

  }
}
#endif
