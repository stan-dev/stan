#ifndef STAN__AGRAD__REV__VAR_STACK_HPP
#define STAN__AGRAD__REV__VAR_STACK_HPP

#include <vector>
#include <stan/memory/stack_alloc.hpp>

namespace stan {
  namespace agrad {

    // forward declaration of chainable
    class chainable;
    class chainable_alloc;
    
    // FIXME: manage all this as a single singleton (thread local)
    extern std::vector<chainable*> var_stack_; 
    extern std::vector<chainable*> var_nochain_stack_; 
    extern std::vector<chainable_alloc*> var_alloc_stack_;
    extern memory::stack_alloc memalloc_;
    
    /**
     * A chainable_alloc is an object which is constructed and destructed normally
     * but the memory lifespan is managed along with the arena allocator for the 
     * gradient calculation.  A chainable_alloc should never be created on the
     * stack, only with a new call.
     */
    class chainable_alloc {
    public:
      chainable_alloc() {
        var_alloc_stack_.push_back(this);
      }
      virtual ~chainable_alloc() { };
    };
    
    /**
     * Recover memory used for all variables for reuse.
     */
    static inline void recover_memory() {
      var_stack_.clear();
      var_nochain_stack_.clear();
      for (size_t i = 0; i < var_alloc_stack_.size(); i++)
        delete var_alloc_stack_[i];
      var_alloc_stack_.clear();
      memalloc_.recover_all();
    }

    /**
     * Return all memory used for gradients back to the system.
     */
    static inline void free_memory() {
      memalloc_.free_all();
    }


  }
}
#endif
