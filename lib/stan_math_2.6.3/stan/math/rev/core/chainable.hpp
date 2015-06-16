#ifndef STAN_MATH_REV_CORE_CHAINABLE_HPP
#define STAN_MATH_REV_CORE_CHAINABLE_HPP

#include <stan/math/rev/core/chainablestack.hpp>
#include <cstddef>

namespace stan {
  namespace math {

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



  }
}
#endif
