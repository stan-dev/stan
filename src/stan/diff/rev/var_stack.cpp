#ifndef __STAN__DIFF__REV__VAR_STACK_CPP__
#define __STAN__DIFF__REV__VAR_STACK_CPP__

#include <stan/diff/rev/var_stack.hpp>

namespace stan {

  namespace diff {

    std::vector<chainable*> var_stack_;
    std::vector<chainable*> var_nochain_stack_;
    std::vector<chainable_alloc*> var_alloc_stack_;
    memory::stack_alloc memalloc_;

  }

}

#endif
