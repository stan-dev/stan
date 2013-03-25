#ifndef __STAN__AGRAD__AGRAD_CPP__
#define __STAN__AGRAD__AGRAD_CPP__

#include "stan/agrad/agrad.hpp"

namespace stan {

  namespace agrad {

    std::vector<chainable*> var_stack_;
    std::vector<chainable*> var_nochain_stack_;
    memory::stack_alloc memalloc_;

  }

}

#endif
