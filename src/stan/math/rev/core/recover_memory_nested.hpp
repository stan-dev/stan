#ifndef STAN__MATH__REV__CORE__RECOVER_MEMORY_NESTED_HPP
#define STAN__MATH__REV__CORE__RECOVER_MEMORY_NESTED_HPP

#include <stan/math/rev/core/chainable.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>
#include <stan/math/rev/core/empty_nested.hpp>
#include <stdexcept>

namespace stan {
  namespace agrad {

    /**
     * Recover only the memory used for the top nested call.  If there
     * is nothing on the nested stack, then a
     * <code>std::logic_error</code> exception is thrown.
     *
     * @throw std::logic_error if <code>empty_nested()</code> returns
     * <code>true</code>
     */
    static inline void recover_memory_nested() {
      if (empty_nested())
        throw std::logic_error("empty_nested() must be false"
                               " before calling recover_memory_nested()");

      ChainableStack::var_stack_.resize(ChainableStack::nested_var_stack_sizes_.back());
      ChainableStack::nested_var_stack_sizes_.pop_back();

      ChainableStack::var_nochain_stack_.resize(ChainableStack::nested_var_nochain_stack_sizes_.back());
      ChainableStack::nested_var_nochain_stack_sizes_.pop_back();

      for (size_t i = ChainableStack::nested_var_alloc_stack_starts_.back();
           i < ChainableStack::var_alloc_stack_.size();
           ++i) {
        delete ChainableStack::var_alloc_stack_[i];
      }
      ChainableStack::var_alloc_stack_.resize(ChainableStack::nested_var_alloc_stack_starts_.back());
      ChainableStack::nested_var_alloc_stack_starts_.pop_back();

      ChainableStack::memalloc_.recover_nested();
    }

  }
}
#endif
