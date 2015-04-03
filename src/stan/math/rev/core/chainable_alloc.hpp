#ifndef STAN__MATH__REV__CORE__CHAINABLE_ALLOC_HPP
#define STAN__MATH__REV__CORE__CHAINABLE_ALLOC_HPP

#include <stan/math/rev/core/chainablestack.hpp>
#include <stdexcept>

namespace stan {
  namespace agrad {


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
      virtual ~chainable_alloc() { }
    };

  }
}
#endif
