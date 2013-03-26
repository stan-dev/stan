#ifndef __STAN__AGRAD__REV__VAR_STACK_HPP__
#define __STAN__AGRAD__REV__VAR_STACK_HPP__

#include <vector>
#include <stan/memory/stack_alloc.hpp>

namespace stan {
  namespace agrad {

    // forward declaration of chainable
    class chainable;

    // FIXME: manage all this as a single singleton (thread local)
    extern std::vector<chainable*> var_stack_; 
    extern std::vector<chainable*> var_nochain_stack_; 
    extern memory::stack_alloc memalloc_;

    /**
     * Recover memory used for all variables for reuse.
     */
    static void recover_memory() {
      var_stack_.clear();
      var_nochain_stack_.clear();
      memalloc_.recover_all();
    }

    /**
     * Return all memory used for gradients back to the system.
     */
    static void free_memory() {
      memalloc_.free_all();
    }

    /**
     * Reset all adjoint values in the stack to zero.
     */
    static void set_zero_all_adjoints() {
      for (size_t i = 0; i < var_stack_.size(); ++i)
        var_stack_[i]->set_zero_adjoint();
      for (size_t i = 0; i < var_nochain_stack_.size(); ++i)
        var_nochain_stack_[i]->set_zero_adjoint();
    }

  }
}
#endif
