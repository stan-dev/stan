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

  }
}
#endif
