#ifndef STAN__AGRAD__REV__VAR_STACK_HPP
#define STAN__AGRAD__REV__VAR_STACK_HPP

#include <stdexcept>
#include <vector>
#include <stan/memory/stack_alloc.hpp>

namespace stan {
  namespace agrad {

    // forward declaration of chainable
    class chainable;
    class chainable_alloc;
    
    extern std::vector<chainable*> var_stack_; 
    extern std::vector<chainable*> var_nochain_stack_; 
    extern std::vector<chainable_alloc*> var_alloc_stack_;
    extern memory::stack_alloc memalloc_;

    // nested positions
    extern std::vector<size_t> nested_var_stack_sizes_;
    extern std::vector<size_t> nested_var_nochain_stack_sizes_;
    extern std::vector<size_t> nested_var_alloc_stack_starts_;
    
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
     * Return true if there is no nested autodiff being executed.
     */
    static inline bool empty_nested() {
      return nested_var_stack_sizes_.empty();
    }

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
      var_stack_.clear();
      var_nochain_stack_.clear();
      for (size_t i = 0; i < var_alloc_stack_.size(); i++)
        delete var_alloc_stack_[i];
      var_alloc_stack_.clear();
      memalloc_.recover_all();
    }
    
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

      var_stack_.resize(nested_var_stack_sizes_.back());
      nested_var_stack_sizes_.pop_back();

      var_nochain_stack_.resize(nested_var_nochain_stack_sizes_.back());
      nested_var_nochain_stack_sizes_.pop_back();

      for (size_t i = nested_var_alloc_stack_starts_.back();
           i < var_alloc_stack_.size(); 
           ++i)
        delete var_alloc_stack_[i];
      nested_var_alloc_stack_starts_.pop_back();

      memalloc_.recover_nested();
    }

    /**
     * Record the current position so that <code>recover_memory_nested()</code>
     * can find it.
     */
    static inline void start_nested() {
      nested_var_stack_sizes_.push_back(var_stack_.size());
      nested_var_nochain_stack_sizes_.push_back(var_nochain_stack_.size());
      nested_var_alloc_stack_starts_.push_back(var_alloc_stack_.size());
      memalloc_.start_nested();
    }

    static inline size_t nested_size() {
      return var_stack_.size() - nested_var_stack_sizes_.back();
    }

  }
}
#endif
