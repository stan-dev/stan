#ifndef STAN__AGRAD__REV__VAR_STACK_HPP
#define STAN__AGRAD__REV__VAR_STACK_HPP

#include <stdexcept>
#include <vector>
#include <stan/memory/stack_alloc.hpp>

namespace stan {
  namespace agrad {

    template<typename ChainableT,
             typename ChainableAllocT>
    struct AutodiffStackStorage {
      static std::vector<ChainableT*> var_stack_;
      static std::vector<ChainableT*> var_nochain_stack_;
      static std::vector<ChainableAllocT*> var_alloc_stack_;
      static memory::stack_alloc memalloc_;

      // nested positions
      static std::vector<size_t> nested_var_stack_sizes_;
      static std::vector<size_t> nested_var_nochain_stack_sizes_;
      static std::vector<size_t> nested_var_alloc_stack_starts_;
    };

    template<typename ChainableT, typename ChainableAllocT>
    std::vector<ChainableT*> AutodiffStackStorage<ChainableT,ChainableAllocT>::var_stack_;
    template<typename ChainableT, typename ChainableAllocT>
    std::vector<ChainableT*> AutodiffStackStorage<ChainableT,ChainableAllocT>::var_nochain_stack_;
    template<typename ChainableT, typename ChainableAllocT>
    std::vector<ChainableAllocT*> AutodiffStackStorage<ChainableT,ChainableAllocT>::var_alloc_stack_;
    template<typename ChainableT, typename ChainableAllocT>
    memory::stack_alloc AutodiffStackStorage<ChainableT,ChainableAllocT>::memalloc_;
    template<typename ChainableT, typename ChainableAllocT>
    std::vector<size_t> AutodiffStackStorage<ChainableT,ChainableAllocT>::nested_var_stack_sizes_;
    template<typename ChainableT, typename ChainableAllocT>
    std::vector<size_t> AutodiffStackStorage<ChainableT,ChainableAllocT>::nested_var_nochain_stack_sizes_;
    template<typename ChainableT, typename ChainableAllocT>
    std::vector<size_t> AutodiffStackStorage<ChainableT,ChainableAllocT>::nested_var_alloc_stack_starts_;

    // forward declaration of chainable
    class chainable;
    class chainable_alloc;

    typedef AutodiffStackStorage<chainable,chainable_alloc> ChainableStack;

    /**
     * A chainable_alloc is an object which is constructed and destructed normally
     * but the memory lifespan is managed along with the arena allocator for the 
     * gradient calculation.  A chainable_alloc should never be created on the
     * stack, only with a new call.
     */
    class chainable_alloc {
    public:
      chainable_alloc() {
        ChainableStack::var_alloc_stack_.push_back(this);
      }
      virtual ~chainable_alloc() { };
    };
    
    /**
     * Return true if there is no nested autodiff being executed.
     */
    static inline bool empty_nested() {
      return ChainableStack::nested_var_stack_sizes_.empty();
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
      ChainableStack::var_stack_.clear();
      ChainableStack::var_nochain_stack_.clear();
      for (size_t i = 0; i < ChainableStack::var_alloc_stack_.size(); i++)
        delete ChainableStack::var_alloc_stack_[i];
      ChainableStack::var_alloc_stack_.clear();
      ChainableStack::memalloc_.recover_all();
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

      ChainableStack::var_stack_.resize(ChainableStack::nested_var_stack_sizes_.back());
      ChainableStack::nested_var_stack_sizes_.pop_back();

      ChainableStack::var_nochain_stack_.resize(ChainableStack::nested_var_nochain_stack_sizes_.back());
      ChainableStack::nested_var_nochain_stack_sizes_.pop_back();

      for (size_t i = ChainableStack::nested_var_alloc_stack_starts_.back();
           i < ChainableStack::var_alloc_stack_.size(); 
           ++i)
        delete ChainableStack::var_alloc_stack_[i];
      ChainableStack::nested_var_alloc_stack_starts_.pop_back();

      ChainableStack::memalloc_.recover_nested();
    }

    /**
     * Record the current position so that <code>recover_memory_nested()</code>
     * can find it.
     */
    static inline void start_nested() {
      ChainableStack::nested_var_stack_sizes_.push_back(ChainableStack::var_stack_.size());
      ChainableStack::nested_var_nochain_stack_sizes_.push_back(ChainableStack::var_nochain_stack_.size());
      ChainableStack::nested_var_alloc_stack_starts_.push_back(ChainableStack::var_alloc_stack_.size());
      ChainableStack::memalloc_.start_nested();
    }

    static inline size_t nested_size() {
      return ChainableStack::var_stack_.size() - ChainableStack::nested_var_stack_sizes_.back();
    }

  }
}
#endif
