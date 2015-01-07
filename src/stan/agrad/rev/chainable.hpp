#ifndef STAN__AGRAD__REV__CHAINABLE_HPP
#define STAN__AGRAD__REV__CHAINABLE_HPP

#include <vector>
#include <stan/agrad/rev/var_stack.hpp>

namespace stan {
  namespace agrad {

    /**
     * Abstract base class for variable implementations that handles
     * memory management and applying the chain rule.
     */
    class chainable {

    public:

      /**
       * Construct a chainable object.  The implementation
       * in this abstract base class is a no-op.
       */
      chainable() { }

      /**
       * Chainables are not destructible and should go on the function
       * call stack or be allocated with operator new.
       */
      virtual ~chainable() {
        // handled automatically
      }

      /**
       * Apply the chain rule to this variable based on the variables
       * on which it depends.  The base implementation in this class
       * is a no-op. 
       */
      virtual void chain() {
      }

      /**
       * Initialize this chainable's adjoint value to make it
       * the dependent variable in a gradient calculation. 
       */
      virtual void init_dependent() {
      }

      /**
       * Set the value of the adjoint for this chainable
       * to its initial value.
       */
      virtual void set_zero_adjoint() {
      }

      /**
       * Allocate memory from the underlying memory pool.  This memory is
       * is managed by the gradient program and will be recovered as a whole.
       * Classes should not be allocated with this operator if they have
       * non-trivial destructors.
       *
       * @param nbytes Number of bytes to allocate.
       * @return Pointer to allocated bytes.
       */
      static inline void* operator new(size_t nbytes) {
        return ChainableStack::memalloc_.alloc(nbytes);
      }

      /**
       * Delete a pointer from the underlying memory pool.  This no-op
       * implementation enables a subclass to throw exceptions in its
       * constructor.  An exception thrown in the constructor of a
       * subclass will result in an error being raised, which is in
       * turn caught and calls delete().
       *
       * See the discussion of "plugging the memory leak" in:  
       *   http://www.parashift.com/c++-faq/memory-pools.html
       */
      static inline void operator delete(void* /* ignore arg */) {
        /* no op */
      }
    };

    
    
    /**
     * Reset all adjoint values in the stack to zero.
     */
    static void set_zero_all_adjoints() {
      for (size_t i = 0; i < ChainableStack::var_stack_.size(); ++i)
        ChainableStack::var_stack_[i]->set_zero_adjoint();
      for (size_t i = 0; i < ChainableStack::var_nochain_stack_.size(); ++i)
        ChainableStack::var_nochain_stack_[i]->set_zero_adjoint();
    }

    /**
     * Compute the gradient for all variables starting from the
     * specified root variable implementation.  Does not recover
     * memory.  This chainable variable's adjoint is initialized using
     * the method <code>init_dependent()</code> and then the chain
     * rule is applied working down the stack from this chainable and
     * calling each chainable's <code>chain()</code> method in turn.
     *
     * <p>This function computes a nested gradient only going back as far
     * as the last nesting.
     *
     * <p>This function does not recover any memory from the computation.
     * 
     * @param vi Variable implementation for root of partial
     * derivative propagation.
     */
    static void grad(chainable* vi) {

      // simple reference implementation (intended as doc):
      //   vi->init_dependent(); 
      //   size_t end = var_stack_.size();
      //   size_t begin = empty_nested() ? 0 : end - nested_size();
      //   for (size_t i = end; --i > begin; )  
      //     var_stack_[i]->chain();

      typedef std::vector<chainable*>::reverse_iterator it_t;
      vi->init_dependent(); 
      it_t begin = ChainableStack::var_stack_.rbegin();
      it_t end = empty_nested() ? ChainableStack::var_stack_.rend() : begin + nested_size();
      for (it_t it = begin; it < end; ++it) {
        (*it)->chain();
      }
    }

  }
}
#endif
