#ifndef __STAN__AGRAD__VAR_STACK_CPP__
#define __STAN__AGRAD__VAR_STACK_CPP__

#include <stan/agrad/var_stack.hpp>

namespace stan {

  namespace agrad {

    std::vector<chainable*> var_stack_;
    std::vector<chainable*> var_nochain_stack_;
    memory::stack_alloc memalloc_;

  }

}

#endif
