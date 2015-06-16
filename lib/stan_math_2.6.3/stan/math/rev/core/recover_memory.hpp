#ifndef STAN_MATH_REV_CORE_RECOVER_MEMORY_HPP
#define STAN_MATH_REV_CORE_RECOVER_MEMORY_HPP

#include <stan/math/rev/core/chainable.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <stan/math/rev/core/empty_nested.hpp>
#include <stdexcept>

namespace stan {
  namespace math {

    /**
     * Recover memory used for all variables for reuse.
     *
     * @throw std::logic_error if <code>empty_nested()</code> returns
     * <code>false</code>
     */
    static inline void recover_memory() {
      if (!empty_nested())
        throw std::logic_error("empty_nested() must be true"
                               " before calling recover_memory()");
      ChainableStack::var_stack_.clear();
      ChainableStack::var_nochain_stack_.clear();
      for (size_t i = 0; i < ChainableStack::var_alloc_stack_.size(); ++i) {
        delete ChainableStack::var_alloc_stack_[i];
      }
      ChainableStack::var_alloc_stack_.clear();
      ChainableStack::memalloc_.recover_all();
    }

  }
}
#endif
